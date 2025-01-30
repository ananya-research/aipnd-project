# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, students first develop code for an image classifier built with PyTorch, then convert it into a command line application.


To run:

cd /workspace/cd0673/43ef1733-cf74-40a9-8c22-b6e440128200/image-classifier-part-1-workspace/home/aipnd-project
python train.py flowers --epochs 10 --gpu

python train.py /workspace/cd0673//43ef1733-cf74-40a9-8c22-b6e440128200/image-classifier-part-1-workspace/home/aipnd-project/flowers --epochs 10 --gpu


python predict.py flowers/test/1/image_06743.jpg checkpoint.pth --top_k 5 --category_names cat_to_name.json --gpu
