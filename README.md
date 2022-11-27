# Brain MRI Image Segmentation
## Winner of the 2021 Imperial College Data Science Challenge
In the [2021 Data Science Institute Computer Vision Challenge](https://www.imperial.ac.uk/data-science/),  we applied semantic image segmentation techniques on magnetic resonance imaging (MRI) to aid the diagnosis of brain tumours. [Prof. Yike Guo](https://www.imperial.ac.uk/people/y.guo), the co-director of the Data Science Institute (DSI), and the DSI judging panel have selected this work as the winner of the "Best Computer Vision Project" award. Since our model achieved the highest accuracy (80%) in the testing tasks, which uses a U-net architecture with a pretrained VGG16 and was trained on 90% of available data.

## Dataset
The dataset of brain MRIs from patients with glioma, examples of MRI images and masks are displayed in this section. The Google Drive URL for the dataset may be found [here](https://drive.google.com/drive/folders/1Y4MUrrfT-Xuos83nOnq8ZWTMZmp9qADH?usp=sharing).
![image](./image&mask.png)

## How to Start 
0. Upload `Dataset` on your Google Drive 
1. Create an account on [Weights & Biases](https://wandb.ai/site) platform (I track my training using `wandb`. `wandb` is a tool for tracking and visualising machine learning experiments in real time. If you have an account on Weights & Biases, all you need to do to get started with this notebook in is copy and paste the API key from your profile) 
2. Open [notebook](https://github.com/xy2119/Brain_MRI_Image_Segmentation/blob/main/MRI_ImageSeg_U_Net_VGG16.ipynb) in Google Colab

## Contributing
If you have any questions or advice towards this repository, feel free to contact me at xy2119@ic.ac.uk.

Any kind of enhancement or contribution is welcomed!
