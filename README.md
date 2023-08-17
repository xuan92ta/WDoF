# Modeling Preference as Weighted Distribution over Functions for User Cold-start Recommendation

This is our PyTorch implementation for the paper:

> Jingxuan Wen, Huafeng Liu and Liping Jing. 2023. Modeling Preference as Weighted Distribution over Functions for User Cold-start Recommendation. In Proceedings of the 32nd ACM International Conference on Information and Knowledge Management.



## Highlights

- To characterize the uncertainty in the user decision process, we model user preference as weighted distribution over functions, with the aid of neural processes.
- To capture the global intent and obtain a more stable learning process, we further consider intra-user uncertainty and inter-user importance respectively.
- We provide a theoretical explanation that why the proposed model performs well than regular neural process based recommendation methods.
- Extensive experiments have been conducted on four wildly used benchmark datasets, demonstrating significant improvements over several state-of-the-art baselines.



## Environment Requirement

The code has been tested running under Python 3.7.10. The required packages are as follows:

- pytorch == 1.4.0
- numpy == 1.20.2
- scipy == 1.6.3
- tqdm == 4.60.0
- bottleneck == 1.3.4
- pandas ==1.3.4



## Example to Run the Codes

The parameters have been clearly introduced in `main.py`. 

- Last.FM dataset

  ```
  python main.py --dataset=lastfm --gpu_id=0 --l_max=50
  ```

- ML 1M dataset

  ```
  python main.py --dataset=ml1m --lr=1e-4 --n_epoch=100
  ```

- Epinions dataset

  ```
  python main.py --dataset=epinions --n_epoch=15
  ```

- Yelp dataset

  ```
  python main.py --dataset=yelp --n_epoch=10
  ```

  

