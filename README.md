# Detecting ship plumes using TROPOMI NO<sub>2</sub>, SO<sub>2</sub> and HCHO observations
## A Master's Thesis by Ruben Ahrens

# Abstract
<!-- % introduction -->
Earth observation is becoming important in gaining insight into shipping emissions which produce a large percentage of global emissions. Maritime pollution contributes to global warming, causes health issues in harbors and coastal regions, and disrupts marine ecosystems. Therefore, the International Maritime Organization (IMO) has imposed strict emission limits, and national governments are restricting emissions in harbors and national waters. Because self-reporting is insufficient for measuring compliance, remote sensing is essential for collecting data on ship emissions in the open sea. In previous works, individual ship plumes have been detected using TROPOMI/Sentinel-5P satellite measurements of NO<sub>2</sub>. This study uses machine learning to investigate the impact of incorporating SO<sub>2</sub> and HCHO measurements, alongside NO<sub>2</sub>, in detecting ships' exhaust plumes. Showing the potential of using gases other than NO<sub>2</sub> in finding anomalously emitting ships, benefitting compliance monitoring.

<!-- % method -->
The study focuses on a portion of the eastern Mediterranean, covering the period from January 2020 to July 2022. From AIS (Automatic Identification System) data, we derived each ship's path for the past two hours until the satellite overpass. We derived the path of the plume from this path and wind data. We trained an XGBoost classifier on data samples from $80km\times 80km$ areas to predict whether ship plumes are present. Using XGBoost, we assess the impact of adding  SO<sub>2</sub> and HCHO on the detectability of plumes. We created subsets by binning the data according to each data sample's estimated total of NO<sub>x</sub> emissions using length and speed from ships (NO<sub>x</sub> proxy). The bins help us study the relationship between the expected ship emission and the plume detectability. We created learning curves where we logarithmically increase the number of samples in the (sub)dataset. This shows if performance can be increased by obtaining more TROPOMI and AIS data.

<!-- % results -->
We present heatmaps that combine learning curves with proxy bins to compare each possible combination of gases (NO<sub>2</sub>, SO<sub>2</sub>, and HCHO). Similar to other experiments, the heatmaps showed an increase in detectability for higher NO<sub>x</sub> proxy bins for gas combinations involving NO<sub>2</sub>.

Comparing the learning curves of different proxy bins shows that for all proxy bins, more data for training the model is beneficial.

We also see the added benefit of SO<sub>2</sub> and HCHO data in learning curves that use the entire tabular database without splitting the data into proxy bins.

<!-- % conclusion -->
Our findings show that incorporating SO<sub>2</sub> and HCHO measurements alongside NO<sub>2</sub> improved the classifier's performance at higher and lower NO<sub>x</sub> proxy values. We also observed SO<sub>2</sub> and HCHO data by themselves to detect ships with a ROC AUC of 0.647 and 0.634 respectively, while showing room for improvement in the learning curves when acquiring more TROPOMI and AIS data. While this performance is worse than NO<sub>2</sub> (ROC AUC 0.684) it shows the value of these gases in ship plume detection.

## Code guide
These are the main files to run to reproduce the results. Other files contain helper functions or data exploration.

Data Acquisition
- `SentHub/catalog.ipynb`: Retrieves a list of relevant TROPOMI files and their metadata.
- `S3/Download_Files.ipynb`: Downloads TROPOMI satellite data from AWS S3.

Data Processing
- `AIS/prep_ais.py`: Processes AIS ship tracking data to calculate ship paths and plume trajectories.
- `make_tab_db/create_agg_db.py`: Creates aggregated database for machine learning.

Machine Learning
- `ML/xgbc.ipynb`: Contains XGBoost classification models, cross-validation, and evaluation metrics.

## BibTex citation
If you use code from this repo, do not forget to cite:

```
@mastersthesis{ahrens2024detecting,
  author       = {Ahrens, Ruben},
  title        = {Detecting ship plumes using TROPOMI NO<sub>2</sub>, SO$_2$ and HCHO observations},
  school       = {Leiden University},
  year         = {2024},
  month        = {1},
  address      = {Leiden, The Netherlands},
  type         = {Master's Thesis in Computer Science},
  note         = {Specialisation: Artificial Intelligence. Supervisors: Prof. F. J. Verbeek and Dr. C. Veenman}
}
```
