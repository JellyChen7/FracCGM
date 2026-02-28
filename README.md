# FracCGM
Spatiotemporal Fluid Flow Prediction in Subsurface Fractured Media using Diffusion Models 
(conditional denoising diffusion implicit models with 2D UNet as basic framework)

[Guodong Chen*](https://scholar.google.com/citations?user=U2YFkAgAAAAJ&hl=zh-TW&oi=ao), 
[Nori Nakata](https://profiles.lbl.gov/88689-nori-nakata), 
Zhongzheng Wang, 
[Zhengfa Bi](https://profiles.lbl.gov/416831-zhengfa-bi), 
[Rie Nakata](https://profiles.lbl.gov/145475-rie-nakata)

---

Here, we present **FracCGM**, a deep-learning framework that leverages denoising diffusion conditional probabilistic models to generatively forecast the evolution of pressure and temperature fields in fractured geothermal reservoirs.

Trained on a diverse library of stochastically generated fracture networks and their corresponding physics-based flow simulations, our model demonstrates:

- High-fidelity probabilistic forecasting
- Generation of probabilistic ensembles
- Quantification of epistemic uncertainty
- Real-time probabilistic prediction capability

FracCGM provides a powerful and efficient paradigm for modeling complex subsurface flows and supports risk-aware geothermal reservoir management.

---

# Network Architecture

![Architecture of FracCGM](Assets/Fig1.jpg)

---

# Case 1  
## Fractured Geothermal System with Three Perpendicular Fractures

![Case1](Assets/Fig2.jpg)

## Prediction Performance Visualization

![Case1 Performance](Assets/Fig3.jpg)

---

# Case 2  
## Fractured Geothermal System with Eight Stochastic Fractures

![Case2](Assets/Fig4.jpg)

## Prediction Performance Visualization

![Case2 Performance](Assets/Fig5.jpg)

## Denoising Process Visualization (20 DDIM Steps)

![Case2 Denoising](Assets/Fig6.jpg)

## Fracture Network Inversion (Ensemble Smoother + Pretrained FracCGM)

![Case2 Inversion 1](Assets/Fig7.jpg)

![Case2 Inversion 2](Assets/Fig8.jpg)

---

# Case 3  

## Scenario 1

![Case3 Scenario 1](Assets/Fig9.jpg)

## Scenario 2

![Case3 Scenario 2](Assets/Fig10.jpg)

---

# Datasets

- `Binary_training.mat`
- `state_training.mat`

---

# Network Training

python main.py

# Autonomous Inverse Fracture Generation
python inference.py

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{Chen2025Diffusion,
  title={Spatiotemporal Fluid Flow Prediction in Subsurface Fractured Media using Diffusion Models},
  author={Guodong, Chen; Nori, Nakata; Zhongzheng, Wang; Zhengfa, Bi; Rie, Nakata},
  year={2026},
}
```
