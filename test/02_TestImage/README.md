### Non-linear reconstruction for a sample BW picture

To run the test, follow the steps below

- In the **demo_csrecovery.py**, set generate to _True_ in order to save the noisy images. 
- Set **csrecover** to _True_ to recover from the noisy images with the same undersampling pattern. 
- Once you have the recovered images, run **test_correlation.py** to generate the correlation plot.