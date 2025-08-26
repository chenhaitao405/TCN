import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.optimize import curve_fit
import warnings

warnings.filterwarnings('ignore')


class MotorAngleAnalyzer:
    """膝关节外骨骼电机角度分析器"""

    def __init__(self, csv_file_path, column_name="knee_angle_l", sampling_rate=100, skip_rows=0, label=None):
        """
        初始化分析器

        参数:
            csv_file_path: CSV文件路径
            column_name: 要分析的列名
            sampling_rate: 采样率（Hz）
            skip_rows: 跳过的行数
            label: 数据标签（用于图表显示）
        """
        self.file_path = csv_file_path
        self.column_name = column_name
        self.sampling_rate = sampling_rate
        self.skip_rows = skip_rows
        self.label = label if label else csv_file_path.split('/')[-1].split('.')[0]

        # 分析结果存储
        self.results = {}
        self.motor_angle = None
        self.time = None

    def load_data(self, filter_label=None, label_column="label"):
        """
        加载数据

        参数:
            filter_label: 如果指定，只保留label列等于该值的数据
            label_column: 标签列的名称
        """
        print(f"\n{'=' * 50}")
        print(f"正在读取文件: {self.file_path}")
        print(f"数据标签: {self.label}")
        print(f"目标列名: {self.column_name}")
        if filter_label is not None:
            print(f"筛选条件: {label_column} = {filter_label}")

        try:
            # 读取CSV文件
            df = pd.read_csv(self.file_path, skiprows=self.skip_rows)
            print(f"原始数据形状: {df.shape}")

            # 清理列名
            df.columns = df.columns.str.strip()

            # 如果需要筛选
            if filter_label is not None:
                if label_column in df.columns:
                    # 筛选数据
                    df_filtered = df[df[label_column] == filter_label]
                    print(f"筛选前数据点: {len(df)}")
                    print(f"筛选后数据点: {len(df_filtered)}")

                    if len(df_filtered) == 0:
                        print(f"警告：没有找到 {label_column}={filter_label} 的数据")
                        raise ValueError(f"筛选后没有数据")

                    df = df_filtered.reset_index(drop=True)
                else:
                    print(f"警告：找不到列 '{label_column}'，跳过筛选")
                    print(f"可用列: {df.columns.tolist()}")

            # 查找目标列
            if self.column_name in df.columns:
                target_col = df[self.column_name]
            else:
                # 忽略大小写匹配
                matching_cols = [col for col in df.columns if col.lower() == self.column_name.lower()]
                if matching_cols:
                    target_col = df[matching_cols[0]]
                    print(f"找到列 '{matching_cols[0]}'")
                else:
                    print(f"错误：找不到列 '{self.column_name}'")
                    print(f"可用列: {df.columns.tolist()}")
                    raise ValueError(f"找不到列 '{self.column_name}'")

            # 数据清理
            if target_col.dtype == 'object':
                target_col = pd.to_numeric(target_col, errors='coerce')

            # 处理缺失值
            nan_count = target_col.isna().sum()
            if nan_count > 0:
                print(f"发现 {nan_count} 个无效值，已处理")
                target_col = target_col.dropna()

            self.motor_angle = target_col.values.astype(float)
            self.time = np.arange(len(self.motor_angle)) / self.sampling_rate

            print(f"有效数据点: {len(self.motor_angle)}")
            print(f"时长: {len(self.motor_angle) / self.sampling_rate:.2f} 秒")

            return True

        except Exception as e:
            print(f"加载数据失败: {e}")
            return False

    def analyze(self):
        """执行完整分析"""
        if self.motor_angle is None:
            if not self.load_data():
                return None

        print(f"\n开始分析 '{self.label}' 数据...")

        # 基础统计
        self.results['max'] = np.max(self.motor_angle)
        self.results['min'] = np.min(self.motor_angle)
        self.results['mean'] = np.mean(self.motor_angle)
        self.results['std'] = np.std(self.motor_angle)
        self.results['peak_to_peak'] = self.results['max'] - self.results['min']
        self.results['amplitude'] = self.results['peak_to_peak'] / 2

        # 去除直流分量
        motor_angle_ac = self.motor_angle - self.results['mean']

        # 峰值检测
        min_peak_height = self.results['amplitude'] * 0.3
        peaks, _ = signal.find_peaks(motor_angle_ac, height=min_peak_height, distance=10)
        valleys, _ = signal.find_peaks(-motor_angle_ac, height=min_peak_height, distance=10)

        self.results['peaks'] = peaks
        self.results['valleys'] = valleys
        self.results['peak_count'] = len(peaks)
        self.results['valley_count'] = len(valleys)

        # 周期分析
        if len(peaks) > 1:
            peak_intervals = np.diff(peaks) / self.sampling_rate
            self.results['avg_period'] = np.mean(peak_intervals)
            self.results['std_period'] = np.std(peak_intervals)
            self.results['avg_frequency'] = 1 / self.results['avg_period']
            self.results['period_cv'] = (self.results['std_period'] / self.results['avg_period']) * 100
        else:
            self.results['avg_period'] = 0
            self.results['std_period'] = 0
            self.results['avg_frequency'] = 0
            self.results['period_cv'] = 0

        # FFT频谱分析
        n = len(motor_angle_ac)
        yf = fft(motor_angle_ac)
        xf = fftfreq(n, 1 / self.sampling_rate)

        positive_freq_idx = xf > 0
        xf_positive = xf[positive_freq_idx]
        yf_positive = np.abs(yf[positive_freq_idx])

        if len(yf_positive) > 0:
            dominant_freq_idx = np.argmax(yf_positive)
            self.results['dominant_frequency'] = xf_positive[dominant_freq_idx]
            self.results['dominant_amplitude'] = yf_positive[dominant_freq_idx] * 2 / n

            # 前3个主频
            top_freq_indices = np.argsort(yf_positive)[-3:][::-1]
            self.results['top_frequencies'] = [(xf_positive[idx], yf_positive[idx] * 2 / n)
                                               for idx in top_freq_indices]

        # 正弦拟合
        try:
            def sine_func(t, A, omega, phi, offset):
                return A * np.sin(omega * t + phi) + offset

            p0 = [self.results['amplitude'],
                  2 * np.pi * self.results['dominant_frequency'],
                  0,
                  self.results['mean']]

            popt, _ = curve_fit(sine_func, self.time, self.motor_angle, p0=p0, maxfev=10000)

            self.results['fitted_amplitude'] = abs(popt[0])
            self.results['fitted_frequency'] = abs(popt[1]) / (2 * np.pi)
            self.results['fitted_phase'] = popt[2]
            self.results['fitted_offset'] = popt[3]

            fitted_curve = sine_func(self.time, *popt)
            ss_res = np.sum((self.motor_angle - fitted_curve) ** 2)
            ss_tot = np.sum((self.motor_angle - self.results['mean']) ** 2)
            self.results['r_squared'] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            self.results['fitted_curve'] = fitted_curve

        except:
            self.results['r_squared'] = 0
            self.results['fitted_curve'] = None

        # 存储频谱数据供绘图
        self.results['freq_data'] = (xf_positive, yf_positive * 2 / n)

        return self.results

    def print_summary(self):
        """打印分析摘要"""
        print(f"\n=== {self.label} 分析结果 ===")
        print(f"角度范围: [{self.results['min']:.2f}°, {self.results['max']:.2f}°]")
        print(f"平均值: {self.results['mean']:.2f}°")
        print(f"标准差: {self.results['std']:.2f}°")
        print(f"幅值: {self.results['amplitude']:.2f}°")
        print(f"主频率: {self.results['dominant_frequency']:.3f} Hz")
        print(f"平均周期: {self.results['avg_period']:.3f} 秒")
        print(f"周期一致性(CV): {self.results['period_cv']:.1f}%")
        if self.results['fitted_curve'] is not None:
            print(f"拟合优度(R²): {self.results['r_squared']:.3f}")


def compare_motor_data(file1, file2, column_name="knee_angle_l", sampling_rate=100,
                       label1=None, label2=None, skip_rows=0, offset2=0,
                       filter_label2=None, label_column="label"):
    """
    对比两个CSV文件的电机角度数据

    参数:
        file1, file2: 两个CSV文件路径
        column_name: 要分析的列名
        sampling_rate: 采样率（Hz）
        label1, label2: 数据标签
        skip_rows: 跳过的行数
        offset2: 第二个文件的角度偏移量（度）
        filter_label2: 第二个文件的筛选条件（只分析label列等于该值的数据）
        label_column: 标签列的名称

    返回:
        comparison_results: 对比结果字典
    """

    print("=" * 60)
    print("膝关节外骨骼电机角度对比分析")
    print("=" * 60)

    # 创建两个分析器
    analyzer1 = MotorAngleAnalyzer(file1, column_name, sampling_rate, skip_rows, label1)
    analyzer2 = MotorAngleAnalyzer(file2, column_name, sampling_rate, skip_rows, label2)

    # 加载数据
    if not analyzer1.load_data():  # 第一个文件不筛选
        print("第一个文件加载失败")
        return None

    # 加载第二个文件（可能需要筛选）
    if not analyzer2.load_data(filter_label=filter_label2, label_column=label_column):
        print("第二个文件加载失败")
        return None

    # 对第二个文件应用角度偏移
    if offset2 != 0:
        print(f"\n应用角度偏移: {analyzer2.label} 的所有角度值减去 {offset2}°")
        analyzer2.motor_angle = analyzer2.motor_angle - offset2
        print(
            f"偏移前范围: [{analyzer2.motor_angle.max() + offset2:.2f}°, {analyzer2.motor_angle.min() + offset2:.2f}°]")
        print(f"偏移后范围: [{analyzer2.motor_angle.max():.2f}°, {analyzer2.motor_angle.min():.2f}°]")

    # 执行分析
    results1 = analyzer1.analyze()
    results2 = analyzer2.analyze()

    if results1 is None or results2 is None:
        print("分析失败，请检查文件和参数")
        return None

    # 打印各自的摘要
    analyzer1.print_summary()
    analyzer2.print_summary()

    # 计算差异
    print("\n" + "=" * 60)
    print("对比分析结果")
    notes = []
    if offset2 != 0:
        notes.append(f"{analyzer2.label} 的原始数据已减去 {abs(offset2)}°")
    if filter_label2 is not None:
        notes.append(f"{analyzer2.label} 只包含 {label_column}={filter_label2} 的数据")
    if notes:
        print(f"(注：{'; '.join(notes)})")
    print("=" * 60)

    comparison = {}

    # 基础统计对比
    print("\n基础统计对比:")
    metrics = ['max', 'min', 'mean', 'amplitude', 'peak_to_peak']
    metric_names = ['最大值', '最小值', '平均值', '幅值', '峰峰值']

    for metric, name in zip(metrics, metric_names):
        val1 = results1[metric]
        val2 = results2[metric]
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 != 0 else 0
        comparison[metric] = {'val1': val1, 'val2': val2, 'diff': diff, 'diff_pct': diff_pct}
        print(f"  {name:8s}: {analyzer1.label:15s} = {val1:8.2f}° | "
              f"{analyzer2.label:15s} = {val2:8.2f}° | "
              f"差异: {diff:+7.2f}° ({diff_pct:+6.1f}%)")

    # 频率对比
    print("\n频率特性对比:")
    freq_metrics = ['dominant_frequency', 'avg_frequency', 'avg_period']
    freq_names = ['主频率', '平均频率', '平均周期']
    units = ['Hz', 'Hz', '秒']

    for metric, name, unit in zip(freq_metrics, freq_names, units):
        val1 = results1[metric]
        val2 = results2[metric]
        diff = val2 - val1
        diff_pct = (diff / val1 * 100) if val1 != 0 else 0
        comparison[metric] = {'val1': val1, 'val2': val2, 'diff': diff, 'diff_pct': diff_pct}
        print(f"  {name:8s}: {analyzer1.label:15s} = {val1:7.3f} {unit:2s} | "
              f"{analyzer2.label:15s} = {val2:7.3f} {unit:2s} | "
              f"差异: {diff:+7.3f} {unit:2s} ({diff_pct:+6.1f}%)")

    # 稳定性对比
    print("\n稳定性对比:")
    print(f"  标准差:   {analyzer1.label:15s} = {results1['std']:7.2f}° | "
          f"{analyzer2.label:15s} = {results2['std']:7.2f}°")
    print(f"  周期CV:   {analyzer1.label:15s} = {results1['period_cv']:7.1f}% | "
          f"{analyzer2.label:15s} = {results2['period_cv']:7.1f}%")

    # 拟合质量对比
    if results1['r_squared'] > 0 and results2['r_squared'] > 0:
        print(f"  拟合R²:   {analyzer1.label:15s} = {results1['r_squared']:7.3f} | "
              f"{analyzer2.label:15s} = {results2['r_squared']:7.3f}")

    # 构建第二个数据集的标签后缀
    label2_suffix = ""
    if filter_label2 is not None:
        label2_suffix += f" ({label_column}={filter_label2})"
    if offset2 != 0:
        label2_suffix += f" (偏移{offset2}°)"

    # 绘制对比图
    fig = plt.figure(figsize=(15, 12))

    # 1. 时域对比
    ax1 = plt.subplot(3, 2, 1)
    t1 = analyzer1.time
    t2 = analyzer2.time
    ax1.plot(t1, analyzer1.motor_angle, 'b-', alpha=0.7, label=analyzer1.label)
    ax1.plot(t2, analyzer2.motor_angle, 'r-', alpha=0.7, label=analyzer2.label + label2_suffix)
    ax1.set_xlabel('时间 (秒)')
    ax1.set_ylabel('角度 (°)')
    ax1.set_title('时域信号对比')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 归一化对比（去除直流分量）
    ax2 = plt.subplot(3, 2, 2)
    motor1_norm = analyzer1.motor_angle - results1['mean']
    motor2_norm = analyzer2.motor_angle - results2['mean']
    ax2.plot(t1, motor1_norm, 'b-', alpha=0.7, label=analyzer1.label)
    ax2.plot(t2, motor2_norm, 'r-', alpha=0.7, label=analyzer2.label + label2_suffix)
    ax2.set_xlabel('时间 (秒)')
    ax2.set_ylabel('归一化角度 (°)')
    ax2.set_title('去直流分量信号对比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 频谱对比
    ax3 = plt.subplot(3, 2, 3)
    xf1, yf1 = results1['freq_data']
    xf2, yf2 = results2['freq_data']
    max_freq = min(10, analyzer1.sampling_rate / 2)
    mask1 = xf1 <= max_freq
    mask2 = xf2 <= max_freq
    ax3.plot(xf1[mask1], yf1[mask1], 'b-', alpha=0.7, label=analyzer1.label)
    ax3.plot(xf2[mask2], yf2[mask2], 'r-', alpha=0.7, label=analyzer2.label + label2_suffix)
    ax3.axvline(results1['dominant_frequency'], color='b', linestyle='--', alpha=0.5)
    ax3.axvline(results2['dominant_frequency'], color='r', linestyle='--', alpha=0.5)
    ax3.set_xlabel('频率 (Hz)')
    ax3.set_ylabel('幅值 (°)')
    ax3.set_title('频谱对比')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 相位图对比
    ax4 = plt.subplot(3, 2, 4)
    vel1 = np.gradient(analyzer1.motor_angle, 1 / analyzer1.sampling_rate)
    vel2 = np.gradient(analyzer2.motor_angle, 1 / analyzer2.sampling_rate)
    ax4.plot(analyzer1.motor_angle, vel1, 'b-', alpha=0.5, label=analyzer1.label)
    ax4.plot(analyzer2.motor_angle, vel2, 'r-', alpha=0.5, label=analyzer2.label + label2_suffix)
    ax4.set_xlabel('角度 (°)')
    ax4.set_ylabel('角速度 (°/s)')
    ax4.set_title('相位图对比')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. 统计对比柱状图
    ax5 = plt.subplot(3, 2, 5)
    metrics_plot = ['amplitude', 'std', 'dominant_frequency']
    metric_labels = ['幅值(°)', '标准差(°)', '主频率(Hz)']
    x = np.arange(len(metrics_plot))
    width = 0.35

    vals1 = [results1[m] for m in metrics_plot]
    vals2 = [results2[m] for m in metrics_plot]

    bars1 = ax5.bar(x - width / 2, vals1, width, label=analyzer1.label, alpha=0.8)
    bars2 = ax5.bar(x + width / 2, vals2, width, label=analyzer2.label + label2_suffix, alpha=0.8)

    ax5.set_xlabel('指标')
    ax5.set_ylabel('值')
    ax5.set_title('关键指标对比')
    ax5.set_xticks(x)
    ax5.set_xticklabels(metric_labels)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')

    # 6. 差异百分比图
    ax6 = plt.subplot(3, 2, 6)
    diff_metrics = ['amplitude', 'mean', 'dominant_frequency', 'std']
    diff_labels = ['幅值', '平均值', '主频率', '标准差']
    diff_values = [comparison[m]['diff_pct'] for m in diff_metrics]

    colors = ['green' if d >= 0 else 'red' for d in diff_values]
    bars = ax6.bar(diff_labels, diff_values, color=colors, alpha=0.7)
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax6.set_ylabel('差异百分比 (%)')
    title2_with_suffix = analyzer2.label + label2_suffix if label2_suffix else analyzer2.label
    ax6.set_title(f'{title2_with_suffix} 相对于 {analyzer1.label} 的变化')
    ax6.grid(True, alpha=0.3, axis='y')

    # 添加数值标签
    for bar, value in zip(bars, diff_values):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{value:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')

    # 添加总标题
    title_parts = ['膝关节外骨骼电机角度对比分析']
    if filter_label2 is not None:
        title_parts.append(f"({analyzer2.label}: {label_column}={filter_label2})")
    if offset2 != 0:
        title_parts.append(f"(偏移{offset2}°)")
    plt.suptitle(' '.join(title_parts), fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # 生成对比报告
    print("\n" + "=" * 60)
    print("分析总结")
    notes = []
    if offset2 != 0:
        notes.append(f"{analyzer2.label} 的数据已应用 {offset2}° 偏移")
    if filter_label2 is not None:
        notes.append(f"{analyzer2.label} 只包含 {label_column}={filter_label2} 的数据")
    if notes:
        print(f"(注：{'; '.join(notes)})")
    print("=" * 60)

    # 判断哪个更好
    score1, score2 = 0, 0

    # 周期一致性（CV越小越好）
    if results1['period_cv'] < results2['period_cv']:
        score1 += 1
        print(f"✓ {analyzer1.label} 的周期一致性更好 (CV: {results1['period_cv']:.1f}% < {results2['period_cv']:.1f}%)")
    else:
        score2 += 1
        print(f"✓ {analyzer2.label} 的周期一致性更好 (CV: {results2['period_cv']:.1f}% < {results1['period_cv']:.1f}%)")

    # 拟合优度（R²越大越好）
    if results1['r_squared'] > results2['r_squared']:
        score1 += 1
        print(f"✓ {analyzer1.label} 的正弦性更好 (R²: {results1['r_squared']:.3f} > {results2['r_squared']:.3f})")
    else:
        score2 += 1
        print(f"✓ {analyzer2.label} 的正弦性更好 (R²: {results2['r_squared']:.3f} > {results1['r_squared']:.3f})")

    # 幅值差异
    amp_diff = abs(results2['amplitude'] - results1['amplitude'])
    if amp_diff > results1['amplitude'] * 0.2:
        print(f"! 幅值差异较大 ({amp_diff:.1f}°, {abs(comparison['amplitude']['diff_pct']):.1f}%)")

    # 频率差异
    freq_diff = abs(results2['dominant_frequency'] - results1['dominant_frequency'])
    if freq_diff > results1['dominant_frequency'] * 0.1:
        print(f"! 频率差异较大 ({freq_diff:.3f} Hz, {abs(comparison['dominant_frequency']['diff_pct']):.1f}%)")

    print(f"\n综合评价: ", end="")
    if score1 > score2:
        print(f"{analyzer1.label} 的运动特性更规律")
    elif score2 > score1:
        print(f"{analyzer2.label} 的运动特性更规律")
    else:
        print("两组数据的运动特性相似")

    return {
        'analyzer1': analyzer1,
        'analyzer2': analyzer2,
        'results1': results1,
        'results2': results2,
        'comparison': comparison
    }


# 使用示例
if __name__ == "__main__":
    # 设置要对比的两个文件
    file1 = "BT24_normal_walk_1_1_1-8_on_exo.csv"  # 第一个文件
    file2 = "kneeData_left.csv"  # 第二个文件（请修改为您的第二个文件名）

    # 参数设置
    column_name = "knee_angle_l"  # 要分析的列名
    sampling_rate = 100  # 采样率（Hz）
    label1 = "测试1"  # 第一个数据的标签
    label2 = "测试2"  # 第二个数据的标签
    skip_rows = 0  # 跳过的行数
    offset2 = 180  # 第二个文件的角度偏移量（减去180度）
    filter_label2 = 2  # 只分析第二个文件中label=2的数据
    label_column = "label"  # 标签列的名称

    try:
        # 执行对比分析
        comparison_results = compare_motor_data(
            file1=file1,
            file2=file2,
            column_name=column_name,
            sampling_rate=sampling_rate,
            label1=label1,
            label2=label2,
            skip_rows=skip_rows,
            offset2=offset2,  # 应用角度偏移
            filter_label2=filter_label2,  # 筛选label=2的数据
            label_column=label_column  # 标签列名称
        )

        if comparison_results:
            print("\n" + "=" * 60)
            print("对比分析完成！")
            print("=" * 60)

            # 可以访问详细结果
            # comparison_results['results1']  # 第一个文件的分析结果
            # comparison_results['results2']  # 第二个文件的分析结果（筛选后）
            # comparison_results['comparison']  # 对比结果

    except Exception as e:
        print(f"\n错误：{e}")
        print("\n请检查：")
        print("1. 两个CSV文件都存在")
        print("2. 列名在两个文件中都存在")
        print("3. 数据格式正确")
        print("4. 如果使用筛选，确保label列存在且有对应的值")