# THESE ARE THE INSTRUCTIONS TO SET UP EVERYTHING. THIS DOCUMENT IS A
# WHOLE TODO thingy
# Setting up datasets, models and packages

### Libraries

`curl -LO https://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl`

`pip install torch-0.2.0.post3-cp27-cp27mu-manylinux1_x86_64.whl`

`vim lib/make.sh faster*/lib/make.sh`

`cd faster*/lib/ && sh make.sh && cd ../../lib && sh make.sh && cd .. && exit 0`

Obsolete pre-trained R-CNN `https://pan.baidu.com/s/1V0QIiEI06tcKQOTcHkaorQ`

Mirror: `https://www.dropbox.com/s/62qxqt477vhb59e/faster_rcnn_1_20_7559.pth?dl=0`

### Bash script for downloading datasets

`cd data/`

To download the VRD dataset: 
`wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip`

Unzip VRD dataset: 
`unzip sg_datset`

Move a file, and convert it from .gif to .jpg: 
`mv sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.gif vrd/4392556686_44d71ff5a0_o.jpg`

Download the annotations: 
`wget http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip`

<!---do we really need this? -->
<!---wget https://drive.google.com/drive/folders/1V8q2i2gHUpSAXTY4Mf6k06WHDVn6MXQ7 -->

For downloading the VGG model trained on ImageNet: 
`wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=0ByuDEGFYmWsbNVF5eExySUtMZmM' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=0ByuDEGFYmWsbNVF5eExySUtMZmM" -O VGG_imagenet.npy && rm -rf /tmp/cookies.txt`

For downloading so\_prior.pkl (I couldn't figure a way to download this through wget, so I downloaded locally) and then just used scp. This has to be put in the `~/data/vrd/ folder`
`scp ./so_prior.pkl findwise@10.10.9.30:/opt/interactionwise/interactionwise-vrd/data/vrd/so_prior.pkl`

For downloading VG dataset:
`wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1QrxXRE4WBPDVN81bYsecCxrlzDkR2zXZ' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1QrxXRE4WBPDVN81bYsecCxrlzDkR2zXZ" -O vg.zip && rm -rf /tmp/cookies.txt`

Download the checkpoint:
`./scripts/gd.pl "https://drive.google.com/uc?id=1_jWnvWNwlJ2ZqKbDMHsSs4BjTblg0FSe&export=download" "models/epoch_4_checkpoint.pth.tar"`

To download the Faster R-CNN pretrained model, we downloaded locally and uploaded to server via scp (in the models folder
Actually, no:
scripts/gd.pl https://drive.google.com/file/d/11YQ7Ctj7kaau6WTx5MKkbw6PIxJAyvsZ/view faster_rcnn_1_20_7559.pth

replace this image in vrd_test
https://raw.githubusercontent.com/GriffinLiang/vrd-dsr/master/data/vrd/4392556686_44d71ff5a0_o.jpg
see Griffin Liang for more:
wget https://raw.githubusercontent.com/GriffinLiang/vrd-dsr/master/data/vrd/4392556686_44d71ff5a0_o.jpg
mv 4392556686_44d71ff5a0_o.jpg data/vrd/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.jpg
rm data/vrd/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.gif

