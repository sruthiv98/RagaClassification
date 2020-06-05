# Name That Raga: Classification and Analysis of Indian Classical Music

## Background
Indian Classical music contains two primary divisions - North Indian Classical (_Hindustani_), and South Indian Classical (_Carnatic_). While both styles have their fundamental differences, the underlying structure of styles can be captured in a _raga/raag/ragam_. A raga is defined as “a pattern of notes having characteristic intervals, rhythms, and embellishments, used as a basis for improvisation.” A raga can be compared to a type of scale in Western classical music. Though Western classical music does not have a direct equivalent to this concept, a raga is somewhat comparable to certain scales, such as a natural harmonic minor or a major scale. Every song has a raga that it is set to. For example, the raga _Kirvani_ is equivalent to the natural harmonic scale in Western music.

Our goal with this project is to quantify ragas in a way that we can then build a raga identification tool. This tool would be able to “listen” to an audio clip and be able to identify the raga that the song is set to. It would mimic what seasoned listeners of Indian classical music do already: try to identify a raga while listening to music. By quantifying the features of a ragam, we will attempt to build a raga identification classifier. 

To manage the scope of this project with the time we have, we will build this tool to be functional for 10 prominent Hindustani/Carnatic ragas. These 10 prominent ragas are known as _thaats_. In Hindustani music, these are known as: _Asavari, Bilawal, Bhairav, Bhairavi, Kafi, Kalyan, Khamaj, Marva, Poorvi,_ and _Todi_. The corresponding ragas in Carnatic are known as _Natabhairavi, Dheerashankarabharanam, Mayamalavagowlai, Hanumatodi, Karaharapriya, Kalyani, Harikhamboji, Gamanashama, Kamarvardhani_, and _Shubhapantuvarali_. 


## How To Run: Example

Our repository includes a small amount of test data that you can run this pipeline on and obtain results. 

1. Clone this repository to have a local copy of these files.
2. On the command line, navigate to this repository locally 
3. The command *python run.py test-project* runs the pipeline with the test-project target. This will load, clean, extract features, and build a model with the small amount of data included in the *testdata_raw* folder. 
4. You should now have a csv for loaded data, cleaned data, and the model that was built. 

## How To Run: Classification of Your Own Audio Files

1. Clone this repository. 
2. On the command line, navigate to this repository locally. 
3. Add your own data to *data/raw*
4. On the command line, the command *python run.py full-project* runs the pipeline with the full-project target. This will load, clean, extract features, and build a model with the data that you have included in the *data/raw* folder.
5. You should now have a csv for loaded data, cleaned data, and the model that was built.
