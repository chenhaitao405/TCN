#include "rknn/rknn_api.h"
#include "model_process/model_process.h"

#include <atomic>
#include <algorithm>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <pthread.h>
#include <sched.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

static std::atomic<bool> g_exit_flag(false);

static const float g_mean_vals[8] = {
	0.0f, 0.0f, 0.0f,
	0.0f, 0.0f, 0.0f,
	0.0f, 0.0f
};
static const float g_std_vals[8] = {
	1.0f, 1.0f, 1.0f,
	1.0f, 1.0f, 1.0f,
	1.0f, 1.0f
};

static void stop_handler(int signum)
{
	printf("\nReceived signal %d, stopping...\n", signum);
	g_exit_flag.store(true);
}

static float read_chip_temperature()
{
	const char* path = "/sys/class/thermal/thermal_zone0/temp";
	std::ifstream file(path);
	if (!file.is_open())
		return -273.15f;

	int millideg = 0;
	file >> millideg;
	if (file.fail())
		return -273.15f;

	return millideg / 1000.0f;
}

static bool try_write_node(const char* path, const char* value)
{
	std::ofstream file(path);
	if (!file.is_open())
		return false;

	file << value;
	if (file.fail())
		return false;

	printf("[perf] write ok: %s <- %s\n", path, value);
	return true;
}

static bool read_node_first_line(const char* path, std::string& out)
{
	std::ifstream file(path);
	if (!file.is_open())
		return false;

	if (!std::getline(file, out))
		return false;

	return true;
}

static void try_write_candidates(const std::vector<std::string>& paths, const std::vector<std::string>& values)
{
	for (const auto& p : paths) {
		for (const auto& v : values) {
			if (try_write_node(p.c_str(), v.c_str())) {
				return;
			}
		}
	}
}

static std::vector<std::string> make_freq_candidates(int target_hz)
{
	std::vector<std::string> out;
	if (target_hz > 0) {
		out.emplace_back(std::to_string(target_hz));
		out.emplace_back(std::to_string(target_hz / 1000000));
	}

	const std::vector<std::string> fallback = {
		"420000000", "420",
		"600000000", "600",
		"800000000", "800",
		"1000000000", "1000",
		"2", "1", "0"
	};
	for (const auto& v : fallback) {
		if (std::find(out.begin(), out.end(), v) == out.end()) {
			out.emplace_back(v);
		}
	}
	return out;
}

static void setup_perf_mode(bool perf_mode, int npu_target_hz)
{
	if (!perf_mode)
		return;

	if (npu_target_hz > 600000000) {
		printf("[perf] warning: requested npu freq %d is high, board may clamp to lower level\n", npu_target_hz);
	}

	const std::vector<std::string> npu_freq_values = make_freq_candidates(npu_target_hz);

	// Best effort: different BSP versions expose slightly different sysfs nodes.
	for (int i = 0; i < 8; ++i) {
		char gov_path[128] = {0};
		snprintf(gov_path, sizeof(gov_path),
				 "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_governor", i);
		try_write_node(gov_path, "performance");
	}

	// Compatible with multiple BSP layouts:
	// 1) devfreq path (fdab0000/ff660000)
	// 2) procfs path (/proc/rknpu)
	// For manual freq control, userspace is usually required.
	try_write_candidates(
		{
			"/sys/class/devfreq/ff660000.npu/governor",
			"/sys/class/devfreq/fdab0000.npu/governor",
			"/sys/devices/platform/ff660000.npu/devfreq/ff660000.npu/governor",
			"/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/governor"
		},
		{"userspace", "performance"}
	);

	try_write_candidates(
		{
			"/sys/class/devfreq/ff660000.npu/min_freq",
			"/sys/class/devfreq/fdab0000.npu/min_freq",
			"/sys/devices/platform/ff660000.npu/devfreq/ff660000.npu/min_freq",
			"/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/min_freq"
		},
		npu_freq_values
	);

	try_write_candidates(
		{
			"/sys/class/devfreq/ff660000.npu/max_freq",
			"/sys/class/devfreq/fdab0000.npu/max_freq",
			"/sys/devices/platform/ff660000.npu/devfreq/ff660000.npu/max_freq",
			"/sys/devices/platform/fdab0000.npu/devfreq/fdab0000.npu/max_freq"
		},
		npu_freq_values
	);

	// Some kernels expose runtime controls in /proc/rknpu
	try_write_candidates(
		{"/proc/rknpu/power"},
		{"on", "1"}
	);

	try_write_candidates(
		{"/proc/rknpu/freq"},
		npu_freq_values
	);

	std::string freq_text;
	if (read_node_first_line("/proc/rknpu/freq", freq_text)) {
		printf("[perf] /proc/rknpu/freq => %s\n", freq_text.c_str());
	}

	std::string load_text;
	if (read_node_first_line("/proc/rknpu/load", load_text)) {
		printf("[perf] /proc/rknpu/load => %s\n", load_text.c_str());
	}
}

static void pin_current_thread(int core_id)
{
	if (core_id < 0)
		return;

	cpu_set_t cpuset;
	CPU_ZERO(&cpuset);
	CPU_SET(core_id, &cpuset);
	pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
}

static void cpu_burn_worker(int worker_id, int pin_core)
{
	pin_current_thread(pin_core);

	volatile uint64_t acc = static_cast<uint64_t>(worker_id + 1) * 1469598103934665603ULL;
	while (!g_exit_flag.load()) {
		for (int i = 0; i < 1000000; ++i) {
			acc ^= (acc << 13);
			acc ^= (acc >> 7);
			acc ^= (acc << 17);
			acc += 0x9E3779B97F4A7C15ULL;
		}
	}

	if (acc == 0xFFFFFFFFFFFFFFFFULL) {
		printf("cpu burn impossible branch\n");
	}
}

static bool ensure_dir_exists(const std::string& dir)
{
	struct stat st;
	if (stat(dir.c_str(), &st) == 0) {
		return S_ISDIR(st.st_mode);
	}
	if (mkdir(dir.c_str(), 0755) == 0) {
		return true;
	}
	if (errno == EEXIST) {
		return true;
	}
	return false;
}

static void flash_stress_worker(
	int worker_id,
	const std::string& flash_dir,
	int chunk_kb,
	int span_mb,
	int fsync_every)
{
	if (!ensure_dir_exists(flash_dir)) {
		printf("[FLASH-%d] create dir failed: %s\n", worker_id, flash_dir.c_str());
		return;
	}

	const size_t chunk_bytes = static_cast<size_t>(std::max(4, chunk_kb)) * 1024;
	const size_t span_bytes = static_cast<size_t>(std::max(1, span_mb)) * 1024 * 1024;
	const int safe_fsync_every = std::max(1, fsync_every);

	std::string file_path = flash_dir + "/flash_stress_" + std::to_string(worker_id) + ".bin";
	int fd = open(file_path.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
	if (fd < 0) {
		printf("[FLASH-%d] open failed: %s errno=%d\n", worker_id, file_path.c_str(), errno);
		return;
	}

	std::vector<uint8_t> buf(chunk_bytes, static_cast<uint8_t>(0xA0 + (worker_id & 0x0F)));
	uint64_t write_count = 0;
	uint64_t bytes_written = 0;
	off_t offset = 0;
	auto t0 = std::chrono::steady_clock::now();

	while (!g_exit_flag.load()) {
		ssize_t n = pwrite(fd, buf.data(), buf.size(), offset);
		if (n != static_cast<ssize_t>(buf.size())) {
			printf("[FLASH-%d] pwrite failed errno=%d\n", worker_id, errno);
			std::this_thread::sleep_for(std::chrono::milliseconds(5));
			continue;
		}

		++write_count;
		bytes_written += static_cast<uint64_t>(n);
		offset += static_cast<off_t>(n);

		if (fsync_every > 0 && (write_count % static_cast<uint64_t>(safe_fsync_every) == 0)) {
			fdatasync(fd);
		}

		if (static_cast<size_t>(offset) + chunk_bytes > span_bytes) {
			offset = 0;
			ftruncate(fd, 0);
		}

		if (write_count % 512 == 0) {
			auto t1 = std::chrono::steady_clock::now();
			double sec = std::chrono::duration<double>(t1 - t0).count();
			double mb = static_cast<double>(bytes_written) / (1024.0 * 1024.0);
			double mbps = (sec > 0.0) ? (mb / sec) : 0.0;
			printf("[FLASH-%d] wrote=%.1fMB avg=%.1fMB/s\n", worker_id, mb, mbps);
		}
	}

	fdatasync(fd);
	close(fd);
	printf("[FLASH-%d] stop\n", worker_id);
}

static void npu_run_loop(
	rknn_context ctx,
	rknn_tensor_mem* input_mem,
	const rknn_tensor_attr& input_attr,
	int print_every)
{
	const int h = input_attr.dims[2];
	const int w = input_attr.dims[3];
	const int elem_count = h * w * 8;

	std::vector<float> inputs(elem_count, 0.25f);
	uint64_t count = 0;
	double total_ms = 0.0;
	auto report_t0 = std::chrono::steady_clock::now();

	while (!g_exit_flag.load()) {
		NCHW_float32_to_NC1HWC2_int8(
			inputs.data(),
			reinterpret_cast<int8_t*>(input_mem->virt_addr),
			1,
			8,
			h,
			w,
			input_attr.scale,
			input_attr.zp,
			g_mean_vals,
			g_std_vals,
			8);

		auto t0 = std::chrono::steady_clock::now();
		int ret = rknn_run(ctx, nullptr);
		auto t1 = std::chrono::steady_clock::now();
		if (ret != RKNN_SUCC) {
			printf("rknn_run failed: %d\n", ret);
			g_exit_flag.store(true);
			break;
		}

		++count;
		total_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

		if (print_every > 0 && count % static_cast<uint64_t>(print_every) == 0) {
			const double avg_ms = total_ms / static_cast<double>(count);
			const double fps = 1000.0 / avg_ms;
			const auto now = std::chrono::steady_clock::now();
			const double sec = std::chrono::duration<double>(now - report_t0).count();
			const float temp = read_chip_temperature();
			printf("[NPU] count=%llu avg=%.3fms fps=%.1f temp=%.2fC elapsed=%.1fs\n",
				   static_cast<unsigned long long>(count), avg_ms, fps, temp, sec);
		}
	}

	if (count > 0) {
		const double avg_ms = total_ms / static_cast<double>(count);
		printf("[NPU] final count=%llu avg=%.3fms fps=%.1f\n",
			   static_cast<unsigned long long>(count), avg_ms, 1000.0 / avg_ms);
	}
}

static void resource_destroy(rknn_context ctx, rknn_tensor_mem* input_mem, rknn_tensor_mem* output_mem)
{
	if (input_mem)
		rknn_destroy_mem(ctx, input_mem);
	if (output_mem)
		rknn_destroy_mem(ctx, output_mem);
	if (ctx)
		rknn_destroy(ctx);
}

int main(int argc, char** argv)
{
	const int cpu_cores = static_cast<int>(std::thread::hardware_concurrency());

	int cpu_threads = (cpu_cores > 0) ? cpu_cores : 2;
	int duration_sec = 0;
	int print_every = 100000;
	int npu_target_hz = 420000000;
	int flash_threads = 1;
	int flash_chunk_kb = 256;
	int flash_span_mb = 64;
	int flash_fsync_every = 1;
	std::string flash_dir = "/root";
	bool perf_mode = true;

	static const struct option long_options[] = {
		{"cpu", required_argument, nullptr, 'c'},
		{"duration", required_argument, nullptr, 'd'},
		{"print", required_argument, nullptr, 'p'},
		{"npu-freq", required_argument, nullptr, 'f'},
		{"flash", required_argument, nullptr, 'F'},
		{"flash-kb", required_argument, nullptr, 'k'},
		{"flash-span-mb", required_argument, nullptr, 's'},
		{"flash-fsync", required_argument, nullptr, 'y'},
		{"flash-dir", required_argument, nullptr, 'D'},
		{"no-perf", no_argument, nullptr, 'n'},
		{"help", no_argument, nullptr, 'h'},
		{0, 0, 0, 0}
	};

	int opt;
	while ((opt = getopt_long(argc, argv, "c:d:p:f:F:k:s:y:D:nh", long_options, nullptr)) != -1) {
		switch (opt) {
		case 'c':
			cpu_threads = std::max(1, atoi(optarg));
			break;
		case 'd':
			duration_sec = std::max(0, atoi(optarg));
			break;
		case 'p':
			print_every = std::max(1, atoi(optarg));
			break;
		case 'f':
			npu_target_hz = std::max(1, atoi(optarg));
			break;
		case 'F':
			flash_threads = std::max(0, atoi(optarg));
			break;
		case 'k':
			flash_chunk_kb = std::max(4, atoi(optarg));
			break;
		case 's':
			flash_span_mb = std::max(1, atoi(optarg));
			break;
		case 'y':
			flash_fsync_every = std::max(1, atoi(optarg));
			break;
		case 'D':
			flash_dir = optarg;
			break;
		case 'n':
			perf_mode = false;
			break;
		case 'h':
			printf("Usage: %s <model_path> [--cpu N] [--duration SEC] [--print N] [--npu-freq HZ] [--flash N] [--flash-kb KB] [--flash-span-mb MB] [--flash-fsync N] [--flash-dir PATH] [--no-perf]\n", argv[0]);
			return 0;
		default:
			printf("Unknown option\n");
			return 1;
		}
	}

	if (optind >= argc) {
		printf("Error: model path is required\n");
		printf("Usage: %s <model_path> [--cpu N] [--duration SEC] [--print N] [--npu-freq HZ] [--flash N] [--flash-kb KB] [--flash-span-mb MB] [--flash-fsync N] [--flash-dir PATH] [--no-perf]\n", argv[0]);
		return 1;
	}

	const char* model_path = argv[optind];
	printf("Model=%s cpu_threads=%d duration=%d npu_freq=%d flash_threads=%d flash_kb=%d flash_span_mb=%d flash_fsync=%d flash_dir=%s perf_mode=%s\n",
		   model_path, cpu_threads, duration_sec, npu_target_hz,
		   flash_threads, flash_chunk_kb, flash_span_mb, flash_fsync_every, flash_dir.c_str(),
		   perf_mode ? "on" : "off");
	if (flash_threads > 0) {
		printf("[FLASH] warning: this test increases flash wear, use only for stress testing\n");
	}

	signal(SIGINT, stop_handler);
	signal(SIGTERM, stop_handler);

	setup_perf_mode(perf_mode, npu_target_hz);

	rknn_context ctx = 0;
	std::vector<rknn_tensor_attr> input_attrs;
	std::vector<rknn_tensor_attr> output_attrs;
	int ret = model_info_parse(ctx, const_cast<char*>(model_path), input_attrs, output_attrs);
	if (ret != RKNN_SUCC) {
		printf("model_info_parse failed: %d\n", ret);
		return 1;
	}

	if (input_attrs.empty() || output_attrs.empty()) {
		printf("Invalid model io attrs\n");
		rknn_destroy(ctx);
		return 1;
	}

	input_attrs[0].pass_through = 1;

	rknn_tensor_mem* input_mem = rknn_create_mem(ctx, input_attrs[0].size_with_stride);
	rknn_tensor_mem* output_mem = rknn_create_mem(ctx, output_attrs[0].size_with_stride);
	if (!input_mem || !output_mem) {
		printf("rknn_create_mem failed\n");
		resource_destroy(ctx, input_mem, output_mem);
		return 1;
	}

	ret = rknn_set_io_mem(ctx, input_mem, &input_attrs[0]);
	if (ret != RKNN_SUCC) {
		printf("rknn_set_io_mem(input) failed: %d\n", ret);
		resource_destroy(ctx, input_mem, output_mem);
		return 1;
	}

	ret = rknn_set_io_mem(ctx, output_mem, &output_attrs[0]);
	if (ret != RKNN_SUCC) {
		printf("rknn_set_io_mem(output) failed: %d\n", ret);
		resource_destroy(ctx, input_mem, output_mem);
		return 1;
	}

	std::vector<std::thread> workers;
	workers.reserve(static_cast<size_t>(cpu_threads));
	for (int i = 0; i < cpu_threads; ++i) {
		int pin_core = (cpu_cores > 0) ? (i % cpu_cores) : -1;
		workers.emplace_back(cpu_burn_worker, i, pin_core);
	}

	std::vector<std::thread> flash_workers;
	flash_workers.reserve(static_cast<size_t>(flash_threads));
	for (int i = 0; i < flash_threads; ++i) {
		flash_workers.emplace_back(
			flash_stress_worker,
			i,
			flash_dir,
			flash_chunk_kb,
			flash_span_mb,
			flash_fsync_every);
	}

	std::thread timer_thread;
	if (duration_sec > 0) {
		timer_thread = std::thread([duration_sec]() {
			std::this_thread::sleep_for(std::chrono::seconds(duration_sec));
			g_exit_flag.store(true);
		});
	}

	npu_run_loop(ctx, input_mem, input_attrs[0], print_every);

	g_exit_flag.store(true);
	if (timer_thread.joinable())
		timer_thread.join();
	for (auto& t : workers) {
		if (t.joinable())
			t.join();
	}
	for (auto& t : flash_workers) {
		if (t.joinable())
			t.join();
	}

	resource_destroy(ctx, input_mem, output_mem);
	return 0;
}
