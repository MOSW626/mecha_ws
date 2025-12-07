chmod 755 setting.sh
echo "Setting up Raspberry Pi..."
sudo apt update
sudo apt install python3
sudo apt update
sudo apt install code
sudo apt install python3-opencv python3-picamera2 python3-gpiozero -y
sudo apt update
sudo apt install python3-venv
python3 -m venv ~/venvs/mecha --system-site-packages
echo ‘source ~/venvs/mecha/bin/activate’ >> ~/.bashrc
source ~/.bashrc
echo "Installing dependencies..."
pip install --upgrade pip
pip install tflite-runtime
curl -L -o Check_dependencies_pi.py https://raw.githubusercontent.com/jiyong-choi/ME203/main/Check_dependencies_pi.py
python Check_dependencies_pi.py
