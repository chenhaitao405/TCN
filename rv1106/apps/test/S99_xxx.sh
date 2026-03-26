touch /etc/init.d/S99_pattern
chmod 777 /etc/init.d/S99_pattern
echo '#!/bin/sh

start() {
        echo "start pattern"
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/pattern_Linux/lib:/oem/usr/lib
        /root/pattern_Linux/power_temp_test /root/pattern_Linux/model/pattern_cnn.rknn &
        echo $! > /var/run/pattern.pid
}
stop() {
        echo "stop pattern"                                                             
        if [ -f /var/run/pattern.pid ]; then
                kill $(cat /var/run/pattern.pid)                                         
                rm /var/run/pattern.pid
        else
                killall power_temp_test
        fi
}
restart() {
        stop
        start
}

case "$1" in
  start)
        start
        ;;
  stop)
        stop
        ;;
  restart|reload)
        restart
        ;;
  *)
        echo "Usage: $0 {start|stop|restart}"
        exit 1
esac

exit $?
' > /etc/init.d/S99_pattern