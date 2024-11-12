# Bridge the Modality and Capability Gaps in Vision-Language Model Selection

🎉The code repository for "Bridge the Modality and Capability Gaps in Vision-Language Model Selection"  in PyTorch. If you use any content of this repo for your work, please cite the following bib entry:

```
  @inproceedings{yic2024swab,
    title={Bridge the Modality and Capability Gaps in Vision-Language Model Selection},
    author={Yi, Chao and He, Yu-Hang and Zhan, De-Chuan and Ye, Han-Jia},
    booktitle={NeurIPS},
    year={2024}
  }
```

## Requirements
### 🗂️ Environment
1. [torch 1.13.0](https://github.com/pytorch/pytorch)
2. [torchvision 0.14.0](https://github.com/pytorch/vision)
3. [open_clip 2.23.0](https://github.com/mlfoundations/open_clip)
4. [sentence-transformers 2.2.2](https://huggingface.co/sentence-transformers)


### 🔎 Model
We use the VLM Zoo in LOVM (originally consisting of 35 models). To increase the diversity of the VLM Zoo, we add several models from the BEiT-3 and BLIP series to the VLM Zoo, ultimately creating a VLM Zoo that contains 43 VLMs. We provide the download link for the VLM Zoo we used, which also includes the MPNet model used in our computation of class relevance. Download link: [link](https://1drv.ms/u/c/211d179501129eff/EQS5-jEfIiJMqdksuMXDveUBQjOPH5gYkFRjyL7mKj8Ggg?e=cYThWB)

After download the expert_knowledge.zip file, you need to unzip this file in `./model` to get the folder `./model/checkpoint`.

### 🔎 Dataset
We use the 23 datasets employed in the LOVM Benchmark. We provide download links for the corresponding datasets. Download link: [link](https://1drv.ms/u/c/211d179501129eff/Ee0bf7DKebRDlUb-KF5suXoBwYnz88N9q8WWlcbGyBMlNg?e=E5RqjB)

After download the datasets.zip file, you need to unzip this file in `./data` to get the folder `./data/datasets`.

### 🔎 PTM(Pre-Trained Model)'s Statistics
You can run the following code to calculate the task relation and the statistics (such as the class-level accuracy of VLMs, Modality Gap Vector, etc.) of VLMs on various datasets.

```
python calc_ptm_stats_in_hist_task.py
python pred_ptm_stats_in_target_task_with_ot.py
bash eval.sh
```
Due to the large number of datasets and models in the LOVM Benchmark, the execution time of the above code may be long. You can also directly download these statistics from the download link we provide:

* ptm_stats: [link](https://1drv.ms/u/c/211d179501129eff/EWQknr1CilNHosxgqq1L4M8Bqq2G5s5TnyZGX9um7wwlew?e=PFzgp0). After download the ptm_stats.zip file, you need to unzip this file to get the folder `./ptm_stats`.
* task_relation: [link](https://1drv.ms/u/c/211d179501129eff/ESDNQ9P5Od5Iszp9l1qk7jABIE7psVbJF7aiM0c3xX9O0w?e=yd2dVB). After download the task_relation.zip file, you need to unzip this file to get the folder `./task_relation`.
* SWAB_inputs: [link](https://1drv.ms/u/c/211d179501129eff/Edho8693kzpBiBjVuUwE1N4BGBz2QwvrDPSc-8CiI5yEHg?e=IAHjij). After download the SWAB_inputs.zip file, you need to unzip this file in `./LOVM` to get the folder `./LOVM/SWAB_inputs`.


## 🔑 Running scripts

If you haven't downloaded PTM's Statistics, you can run the following program:

```
bash run.sh
```
Else you can run:
```
bash eval.sh
python analysis_result.py
```


## 👨‍🏫 Acknowledgment

We would like to express our gratitude to the following repositories for offering valuable components and functions that contributed to our code.

- [LOVM](https://github.com/orrzohar/LOVM)
- [open_clip](https://github.com/mlfoundations/open_clip)
- [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3)
- [BLIP](https://github.com/salesforce/BLIP)