# FracCGM
Spatiotemporal Fluid Flow Prediction in Subsurface Fractured Media using Diffusion Models  
(Conditional Denoising Diffusion Implicit Models with 2D UNet as Basic Framework)

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

```bash
python main.py






# FracCGM
Spatiotemporal Fluid Flow Prediction in Subsurface Fractured Media using Diffusion Models (conditional denoising diffusion implicit models with 2D UNet as basic framework)

[Guodong Chen*](https://scholar.google.com/citations?user=U2YFkAgAAAAJ&hl=zh-TW&oi=ao), [Nori Nakata](https://profiles.lbl.gov/88689-nori-nakata), Zhongzheng Wang, [Zhengfa Bi](https://profiles.lbl.gov/416831-zhengfa-bi), & [Rie Nakata](https://profiles.lbl.gov/145475-rie-nakata)

Here, we present FracCGM, a deep-learning framework that leverages denoising diffusion conditional probabilistic models to generatively forecast the evolution of pressure and temperature fields in fractured geothermal reservoirs. Trained on a diverse library of stochastically generated fracture networks and their corresponding physics-based flow simulations, our model demonstrates the dual capacity for high-fidelity probabilistic forecasting and the generation of probabilistic ensembles. These ensembles effectively encapsulate the epistemic uncertainty arising from stochastic fracture geometries, providing a robust pathway for real-time probabilistic prediction and informed reservoir management. Our results establish FracCGM as a powerful and efficient paradigm for modeling complex subsurface flows, offering significant potential for risk assessment and reservoir management strategies in fractured systems.

## Network architecture:
![Architecture of GenFrac](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig1.jpg "Architecture of FracCGM")

## Case 1: Fractured geothermal system with three perpendicular fractures:
![Case1 of FracCGM](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig2.jpg "Case1 of FracCGM")

## Case 1: Prediction performance visualization of fractured geothermal system with three perpendicular fractures:
![Case1 performance](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig3.jpg "Case1 performance")

## Case 2: Fractured Geothermal System with Eight Stochastic Fractures:
![Case2 of FracCGM](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig4.jpg "Case2 of FracCGM")

## Case 2: Prediction performance visualization of fractured geothermal system with eight perpendicular fractures:
![Case2 performance](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig5.jpg "Case2 performance")

## Case 2: Denoising process visualization with 20 steps via DDIM:
![Case2 denoising](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig6.jpg "Case2 denoising")

## Case 2: Fracture network inversion using ensemble smoother with pretrained FracCGM as forward model:
![Case2 fracture inversion](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig7.jpg "Case2 fracture inversion")

## Case 2: Fracture network inversion using ensemble smoother with pretrained FracCGM as forward model:
![Case2 fracture inversion](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig8.jpg "Case2 fracture inversion")

## Case 3: First scenario of case 3: Prediction performance visualization of fractured geothermal system with various stochastic fractures:
![Case3 of FracCGM](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig9.jpg "Case3 of FracCGM")

## Case 3: Second scenario of case 3: Prediction performance visualization of fractured geothermal system with various stochastic fractures:
![Case3_2 of FracCGM](https://github.com/JellyChen7/FracCGM/raw/master/Assets/Fig10.jpg "Case3_2 of FracCGM")

## Datasets
Binary_training.mat;
state_training.mat

## Network Training
python main.py

## Autonomous Inverse Fracture Generation
python inference.py

If you find the paper or this repository helpful in your publications, please consider citing it.

```bibtex
@article{Chen2025Diffusion,
  title={Spatiotemporal Fluid Flow Prediction in Subsurface Fractured Media using Diffusion Models},
  author={Guodong, Chen; Nori, Nakata; Zhongzheng, Wang; Zhengfa, Bi; Rie, Nakata},
  year={2026},
}
```
