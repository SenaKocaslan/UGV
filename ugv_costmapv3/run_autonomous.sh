#!/usr/bin/env bash
# run_autonomous.sh — TAM OTONOM: ZED + YOLO + costmap + planlayıcı + MOTOR.
# Planlayıcının seçtiği yayın eğriliği differential PWM'e çevrilip motorlara
# gönderilir. Geçerli yay yoksa araç DURUR (planner stop).
#
# Kullanım:
#   ./run_autonomous.sh                 # /dev/ttyCH341USB0
#   ./run_autonomous.sh /dev/ttyUSB0
#
# !!! İLK DENEMEDE config.py'da BASE_PWM'i düşük tut, açık alanda test et !!!
set -e
PORT="${1:-/dev/ttyCH341USB0}"
cd "$(dirname "$0")"
echo "Otonom sürüş: source=live, drive=ON, port=$PORT"
echo "Durdurmak için: pencerede 'q'/ESC ya da Ctrl+C (motorlar durur)."
exec python3 main.py --source live --drive --port "$PORT"
