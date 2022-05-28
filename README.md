# Tiny Faces Detector
 
This repository provides the simplified script to use the tiny faces detector Pytorch model done by [varunagrawal](https://github.com/varunagrawal/tiny-faces-pytorch) to count the number of tiny faces within a given image. The Pytorch implementation by varunagrawal is based on the work done by **Peiyun Hu** & **Deva Ramanan** in their publication [Finding Tiny Faces](https://arxiv.org/abs/1612.04402). 

CLone the repo and run the following command to count the number of tiny faces within an input image.

> python detect.py --image <path_to_image>

detect.py loads the pre-trained tiny faces detector Pytorch model (checkpoint_50_best.pt) that is recommended to be situated within the _/models_ folder. Should checkpoint_50_best.pt not be found in the _/models_ folder, detect.py will download a remote copy from Google Drive.

## Sample Outputs

Original             |  Processed
:-------------------------:|:-------------------------:
[Source](https://c8.alamy.com/comp/D0W7EG/crowds-of-people-at-orchard-road-singapore-D0W7EG.jpg) ![Orchard-1](https://user-images.githubusercontent.com/6497242/170828917-b9b7e78c-2a24-4b6a-a897-814f26f778dd.jpg)  |  ![Orchard-1_processed](https://user-images.githubusercontent.com/6497242/170828872-cc811580-19e2-4f3c-a302-c9c869d47d9c.jpg)
[Source](https://www.pa.gov.sg/images/default-source/module/news/mark-singapore's-52nd-birthday-with-pride-at-our-community-national-day-celebrations)![National_day](https://user-images.githubusercontent.com/6497242/170829038-47f6ce36-eca4-4e26-af5c-6c8a1cf04c79.jpg)  |  ![National_day_processed](https://user-images.githubusercontent.com/6497242/170829042-b201dab7-2a6a-49a8-b120-357edd9f209f.jpg)
[Source](https://i.ytimg.com/vi/9fiz_1q3yTc/maxresdefault.jpg)![big-group-photo](https://user-images.githubusercontent.com/6497242/170829238-811f202d-2268-4f84-bf4b-1bbafe9e403b.jpg) | ![big-group-photo_processed](https://user-images.githubusercontent.com/6497242/170829257-14004bbf-3e49-4e25-abb7-e3f7da5c61d8.jpg)
