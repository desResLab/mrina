### Non-linear reconstruction of arbitrary images

The *csrecovery.py* script recontructs a complex image using various approaches, i.e.

- CS Iterative thresholding
- CS Iterative thresholding + Debiasing
- Orthogonal Matching Pursuit

To run the code, type

```
python3 csrecovery.py fileName uRatio NoiseInt
```
where:
- **fileName** - is the name of the image file. Both image dimensions should be a power of 2 for orthogonality in the wavelet transform.
- **uRatio** - undersampling ratio. This is the ratio of the available k-space frequency measurements.
- **NoiseInt** - noise intensity. This is used to set the noise standard deviation to sigma = NoiseInt * imNrmAvg, where imNrmAvg is the average norm in image space.
