# Model Components

###### [Progress] January 6, 2024
###### [Progress] Elaborated notes from [Little Book of Deep Learning ](https://fleuret.org/public/lbdl.pdf?fbclid=IwAR3jmeQf1k6Q6Qbp6fDmEtklfqo3XMNrHSoIE_2m8By8cpF2sPZjghuq-Zg)

## What is a layer?
### Linear Layers
"Linear" in deep learning != [linear in math](https://en.wikipedia.org/wiki/Linearity#:~:text=In%20mathematics%2C%20a%20linear%20map,(x)%20for%20all%20%CE%B1.). The linear in layers, refers to an [affine operation](https://youtu.be/E3Phj6J287o?si=YW0ya5B9iY3OtiQb) (linear transformation): 

$$ y = Ax + b $$

Where: 
- y = is the transformed vector.
- A = is a linear transformation matrix.
- x = is the original vector.
- b = is a translation vector.

### Dense Layers/fully connected layers
The dense layer is simple affine transformation that can work with input/outputs from multi-dimensional tensors.

<!--START OF FOOTER-->
<hr style="margin-top:9px;height:1px;border: 0;background-image: linear-gradient(to right, rgba(0, 0, 0, 0.0), rgba(0, 0, 0, 0.5),rgba(0, 0, 0, 0.0));">
<!--START OF ISSUE NAVIGATION LINKS-->
<!--START OF ISSUE NAVIGATION LINKS-->
<!--END OF FOOTER-->
