# Problem Statement

The objective of this problem is to be able to detect logos from different documents and know the organization the logos belong to. Example document here is a Bank statement.

As part of this challenge we would like you to develop a learning  algorithm that can take an input logo, and determine the bank it belongs to. These logos can be of different scale, size, quality and image types, so  keep that in consideration while thinking about your approach and  solution.

**Primary Challenge:** Determining which organization the logo belongs to

# Inputs

We have provided some some sample images in the directory `training_images`. These are a mix of different types of images including just logos as well as bank statement headers. We encourage use your own training data as well as you find suitable.

# Expected output.

There are 2 outputs we expect for this problem

1. A document listing out your approach and different algorithms considered. Let us know why you picked an approach and what kind of tweaking needed to be done. If you preferred one algorithm over another it would be very helpful if you could provide your reasoning. Walk us through how you ended up picking your solution. Any accuracy numbers would also be useful.

2. We should be able to run a script to determine which organization an input logo file/document belongs to.

   Example

   ```
   python logo_detection.py bank_of_america.jpg
   # OR
   python logo_detection.py bank_of_america_statement_page_1.jpg
   ```

   ### Sample Output

   ```
   Document belongs to organization: "Bank Of America"
   
   Although `python` is used above, fell free to use any language/framework you like, as long as you can cleanly
   document your solution as mentioned in point 1 above.
   
   A step by step guide on how to run your scripts definitely gets some bonus points :)
   ```

We are expecting outputs to **only** belong to the following organizations:

```
['Bank Of America', 'Wells Fargo', 'JPMorgan Chase', 'Citigroup Inc', 'Capital One', 'Other']
```

Any other logo you can categorize into the "other" category.

# Code on conduct

Please submit original work. Using another persons work as one's own  for these submissions will result in automatic disqualification.