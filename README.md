# UCT: Unbiased Feature Learning with Causal Intervention for Visible-Infrared Person Re-identification
Pytorch code for "Unbiased Feature Learning with Causal Intervention for Visible-Infrared Person Re-identification"

## Updates

 **[23.08.16]** We publish the code for testing, and after the paper is accepted, we will publish the training code.

## Results

| Datasets   | Settings         | Pretrained | Rank@1 | mAP    | mINP   | Model          |
| ---------- | ---------------- | ---------- | ------ | ------ | ------ | -------------- |
| #SYSU-MM01 | All-Search       | ImageNet   | 79.28% | 75.88% | 63.76% | available soon |
| #SYSU-MM01 | Indoor-Search    | ImageNet   | 84.61% | 85.24% | 82.08% | available soon |
| #RegDB     | Visible2Infrared | ImageNet   | 95.29% | 95.87% | 95.23% | available soon |
| #RegDB     | Infrared2Visible | ImageNet   | 92.91% | 93.37% | 92.25% | available soon |

## Data Preparation

- (1) RegDB Dataset: The RegDB dataset can be downloaded from this [website](http://dm.dongguk.edu/link.html) by submitting a copyright form.

    - (Named: "Dongguk Body-based Person Recognition Database (DBPerson-Recog-DB1)" on their website). 

    - A private download link can be requested in [Github](https://github.com/mangye16/Cross-Modal-Re-ID-baseline). 
  
- (2) SYSU-MM01 Dataset: The SYSU-MM01 dataset can be downloaded from this [website](http://isee.sysu.edu.cn/project/RGBIRReID.htm).

   - run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.

## Evaluation

Test a model on SYSU-MM01 or RegDB dataset by 

  ```bash
python test_mine.py --mode all --model_path 'downloaded/checkpoint/path/' --resume 'sysu_all_mAP_best.t' --gpu 1 --dataset sysu
  ```
  - `--dataset`: which dataset "sysu" or "regdb".
  - `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).
  - `--trial`: testing trial (only for RegDB dataset).
  - `--model_path`: the saved model path.
  - `--resume`: the saved model name.
  - `--gpu`:  which gpu to run.

### Usage

Our code extends the pytorch implementation of HCT [1] in [Github](https://github.com/hijune6/Hetero-center-triplet-loss-for-VT-Re-ID).
