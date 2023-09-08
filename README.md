# CombTR
CombTR is an ensemble model developed to automatically segment 13 organs based off of 3D CT scans.
The following models are used: 

* UNETR
* SwinUNETR
* SegResNet

CombTR uses a "stacking" model architecture, where each model's output is the input into a meta-learner, which learns from the ensemble's mistakes and corrects them over time.
This model beat previous research from NVIDIA researchers and others by 0.5-2% in DICE score, respectively.

# Winner of Solano County Science Fair 2023
# Winner of the UCLA Brain Research Institute Award (California State Science and Engineering Fair 2023)
