# Interactionwise -- Semantic Awareness for Visual Relationship Detection

This repo contains the code for our thesis work on Visual Relationship Detection.
It is now quite messy, and we are in the process of tidying it up.
We based our work on [[1]](#1).

## In the meantime... partial instructions to set up some assets (datasets, models and packages)

### Packages

```
pip install -r requirements.txt
pip install git+https://github.com/Infinidat/munch
```

<!---
Obsolete pre-trained R-CNN `https://pan.baidu.com/s/1V0QIiEI06tcKQOTcHkaorQ`

Mirror: `https://www.dropbox.com/s/62qxqt477vhb59e/faster_rcnn_1_20_7559.pth?dl=0`
-->

### VGG16 feature extractor trained on ImageNet:
```
./scripts/gd.pl "https://drive.google.com/file/d/0ByuDEGFYmWsbNVF5eExySUtMZmM/view" "data/VGG_imagenet.npy"
```

### VRD dataset:
```
cd data/vrd
wget http://imagenet.stanford.edu/internal/jcjohns/scene_graphs/sg_dataset.zip
unzip -n sg_datset
cd ../..
```

```
mkdir data/embeddings
wget "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
mv GoogleNews-vectors-negative300.bin.gz data/embeddings
```


### VG dataset:
Please download everything you need from https://visualgenome.org/api/v0/api_home.html into `data/vg`

<!--
`./scripts/gd.pl "https://drive.google.com/file/d/1_jWnvWNwlJ2ZqKbDMHsSs4BjTblg0FSe/view" "models/epoch_4_checkpoint.pth.tar"`
`./scripts/gd.pl "https://drive.google.com/file/d/1e1agFQ32QYZim-Vj07NyZieJnQaQ7YKa/view" "data/vrd/so_prior.pkl"`-->

### Faster R-CNN pre-trained model:
```
./scripts/gd.pl https://drive.google.com/file/d/11YQ7Ctj7kaau6WTx5MKkbw6PIxJAyvsZ/view "faster-rcnn/faster_rcnn_1_20_7559.pth"
```

<!--
./scripts/gd.pl https://drive.google.com/file/d/1QrxXRE4WBPDVN81bYsecCxrlzDkR2zXZ/view vg.zip
unzip vg.zip
rm vg.zip
mv vg data/vg/dsr
-->

<!--Download the annotations:
`wget http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip`-->

<!---do we really need this? -->
<!---wget https://drive.google.com/drive/folders/1V8q2i2gHUpSAXTY4Mf6k06WHDVn6MXQ7 -->

<!--
For downloading so\_prior.pkl This has to be put in the `~/data/vrd/ folder`
`scp  data/vrd/so_prior.pkl`

For downloading VG dataset:
-->

## References
<a id="1">[1]</a>
Kongming Liang, Yuhong Guo, Hong Chang, and Xilin Chen (2018).
	Visual relationship detection with deep structural ranking
https://github.com/GriffinLiang/vrd-dsr

<!--
## References
<a id="1">[1]</a>
Dijkstra, E. W. (1968).
	Go to statement considered harmful.
Communications of the ACM, 11(3), 147-148.
-->

<!--
replace this image in vrd_test
https://raw.githubusercontent.com/GriffinLiang/vrd-dsr/master/data/vrd/4392556686_44d71ff5a0_o.jpg
see Griffin Liang for more:
wget https://raw.githubusercontent.com/GriffinLiang/vrd-dsr/master/data/vrd/4392556686_44d71ff5a0_o.jpg
mv 4392556686_44d71ff5a0_o.jpg data/vrd/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.jpg
rm data/vrd/sg_dataset/sg_test_images/4392556686_44d71ff5a0_o.gif
-->
