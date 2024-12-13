# GCS: Granular Convolution Synthesis

## Try GCS out
### 1. Installation
- [ ] conda create --name gcs python=3.12
- [ ] conda activate gcs
- [ ] pip install -r requirements.txt

### 2. Running the script
- [ ] python granular_convolution_synthesis.py
- [ ] Open the link in your browser
- [ ] Import two audio files and adjust sliders and submit
- [ ] Generated audio can be played and downloaded
Note: Audio generation takes time (depends on the length of your audio as well as sampling rate & your device); for reference, an audio clip of 3-minutes (sr=44,100) may take up to a minute computation time.

## Credits
### Ideas and motivation: 
*[Prof. Carmine Cella](http://www.carminecella.com/)* of UC Berkeley CNMAT

-- Thank you Professor Cella for giving me ideas and helping me explore more of this concept.

### Granular synthesis:
- [Basics of GS](https://cmtext.indiana.edu/synthesis/chapter4_granular.php)

- [Josh Stovall's blog post of granular synthesis and interviews of original paper](https://joshstovall.com/writing/granular-synthesis/)

## Further ideas for improvement
- [ ] Chained convolution effects
- [ ] Real time sliders that loops (i.e. faster implementation of convolution)
- [ ] Better UI
