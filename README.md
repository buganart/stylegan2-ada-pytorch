# Modify StyleGAN2-ADA PyTorch For Colab Training

This repository is forked from [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) that implements the StyleGAN2-ADA based on the paper below.

**Training Generative Adversarial Networks with Limited Data**<br>
Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, Timo Aila<br>
https://arxiv.org/abs/2006.06676<br>

For more details about StyleGAN2-ADA, please go to [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) README.

The purpose of this repository is to modify StyleGAN2-ADA to be able to train and generate using colab notebook, which make the model more accessible to everyone and make artists easier to try and use the model for their artistic works.


# Getting started

As the repository does not modify any of the main modules of the model, the commands list in the README **Getting started** section in the original repository [NVlabs/stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch) should work.

In this repository, colab notebooks are added for the training and generating functionality in the colab environment. 
Wandb is used to keep track of all experiment data.

## [stylegan2ada_train.ipynb](https://github.com/buganart/stylegan2-ada-pytorch/blob/main/stylegan2ada_train.ipynb)

User need to have their data load to Google Drive, and have Drive mounted to the notebook.

Then, in the notebook, set:
1. database_dir (google drive path to dataset)
2. out_dir (google drive path to folder to store experiment data) 
3. resume_id (wandb id if resume run, or empty for new run)
4. resolution (size of image, for example 256,512,1024)
    
After that, run the notebook, and the model will start training. Remember to record the 8 characters wandb id for resume run or for generation.


## [stylegan2ada_generate.ipynb](https://github.com/buganart/stylegan2-ada-pytorch/blob/main/stylegan2ada_generate.ipynb)

User need to have their data load to Google Drive, and have Drive mounted to the notebook.

To load the trained model pkl file, set either: 
* resume_id (trained from the notebook above. The pkl file will be downloaded from wandb and load model), or
* ckpt_file (Google Drive path directly to the pkl file)

Then, set generate parameters for the generate script:
1. seeds (List of random seeds to convert into latent vectors)
2. truncation_psi
3. class_idx (Class label (unconditional if not specified))
4. noise_mode 
5. out_dir (google drive path to folder to store generated data)

Apart from the parameters already in the generate script, some modifications have been done to perform latent space exploration. For a pair of seeds, 2 seed latent vectors will be generated. Then, n latent vectors will be produced using np.linspace with start and end as the seed latent vectors.
6. latent (whether to use latent space exploration or random generation) 
7. frames (If latent, the number of latent vectors for the np.linspace. If not latent, number of random vectors for each seed)

After that, run the notebook, and the generated images will be stored in the out_dir.

## Extra: [api.py](https://github.com/buganart/stylegan2-ada-pytorch/blob/main/api.py)

This is an flask API server only for the exploration/testing purpose.
The user need to download and save pkl file near the api.py script in the following structure.
* stylegan2_repo
    *  api.py
    *  checkpoint
        *   ffhq.pkl
        *   your_ckpt.pkl

Then, run 

    # install dependency
    pip install numpy scipy ninja pyspng wandb flask torch pillow
    # run api
    python api.py
    
After the server is started, construct API request with the following information:

    POST http://127.0.0.1:8080/generate-latent-walk
    "Content-Type: application/json"
    {
        "ckpt_file" : "your_ckpt.pkl", 
        "seeds": "3,7,10",
        "frames": "10",
        "psi": "1",
        "class_idx": "0",
        "noise_mode": "const"
    } 

The `ip_address` should be `127.0.0.1` for windows by default, and `0.0.0.0` for linux by default.
Then, the server will send back a mjpeg stream with latent space walk loop starting from seed 3 to seed 10.
