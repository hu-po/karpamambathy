nvidia-smi
echo "---- python3 --version"
python3 --version
echo "---- nvcc --version"
nvcc --version
echo "---- torch.__version__"
python3 -c "import torch; print(torch.__version__)"
echo "---- torch.cuda.is_available()"
python3 -c "import torch; print(torch.cuda.is_available())"