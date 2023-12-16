import numpy 
import cv2
import matplotlib.pyplot as pl


def read_bin(file_path, size):
    with open(file_path, 'rb') as file:
        data = numpy.fromfile(file, dtype=numpy.uint8, count=size*size)
        return numpy.reshape(data, (size, size))


def stretch(image):
    return (image - numpy.min(image)) / (numpy.max(image) - numpy.min(image)) * 255.0


file_path = '../image/salesmanbin.sec'
size = 256
X = read_bin(file_path, size)

pl.figure(figsize=(12, 6))
pl.subplot(1, 2, 1)
pl.imshow(X, cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('Original Image', fontsize=18)

X2 = numpy.zeros((262, 262))
X2[4:260, 4:260] = X
Y2 = numpy.zeros((262, 262))
for row in range(4, 261):
    for col in range(4, 261):
        Y2[row, col] = numpy.sum(X2[row-3:row+4, col-3:col+4]) / 49

Y = stretch(Y2[4:260, 4:260])

pl.subplot(1,2, 2)
pl.imshow(Y, cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('Filtered Image', fontsize=18)

pl.show()
#--------------------------b-----------------------

pl.figure(figsize=(12, 6))
pl.subplot(2, 4, 1)
pl.imshow(X, cmap='gray')
pl.title('Original image', fontsize=12)
pl.axis('image')
pl.axis('off')

Padsize = 256 + 128 - 1
ZPX = numpy.zeros((Padsize, Padsize))
ZPX[:256, :256] = X

pl.subplot(2, 4, 2)
pl.imshow(ZPX, cmap='gray')
pl.title('Zero Padded', fontsize=12)
pl.axis('image')
pl.axis('off')

H = numpy.zeros((128, 128))
H[62:69, 62:69] = 1 / 49

ZPH = numpy.zeros((Padsize, Padsize))
ZPH[:128, :128] = H

pl.subplot(2, 4, 3)
pl.imshow(ZPH, cmap='gray')
pl.title('Zero Padded Impulse Resp', fontsize=12)
pl.axis('image')
pl.axis('off')

ZPXtilde = numpy.fft.fft2(ZPX)
ZPHtilde = numpy.fft.fft2(ZPH)

ZPXtildeDisplay = numpy.log(1 + numpy.abs(numpy.fft.fftshift(ZPXtilde)))

pl.subplot(2, 4, 4)
pl.imshow(ZPXtildeDisplay, cmap='gray')
pl.title('Log-mag spectrum zero pad', fontsize=12)
pl.axis('image')
pl.axis('off')

ZPHtildeDisplay = numpy.log(1 + numpy.abs(numpy.fft.fftshift(ZPHtilde)))
pl.subplot(2, 4, 5)
pl.imshow(ZPHtildeDisplay, cmap='gray')
pl.title('Log-magnitude spectrum H', fontsize=12)
pl.axis('image')
pl.axis('off')

ZPYtilde = ZPXtilde * ZPHtilde
ZPY = numpy.fft.ifft2(ZPYtilde)

ZPYtildeDisplay = numpy.log(1 + numpy.abs(numpy.fft.fftshift(ZPYtilde)))
pl.subplot(2, 4, 6)
pl.imshow(ZPYtildeDisplay, cmap='gray')
pl.title('Log-magnitude spectrum of result', fontsize=12)
pl.axis('image')
pl.axis('off')

pl.subplot(2, 4, 7)
pl.imshow(numpy.real(ZPY), cmap='gray')
pl.title('Zero Padded Result', fontsize=12)
pl.axis('image')
pl.axis('off')

Y = numpy.real(ZPY[64:320, 64:320])
pl.subplot(2, 4, 8)
pl.imshow(Y, cmap='gray')
pl.title('Final Filtered Image', fontsize=12)
pl.axis('image')
pl.axis('off')
pl.show()

#-------------- c---------------------

def stretch(x):
    xMax = numpy.max(x)
    xMin = numpy.min(x)
    scale_factor = 255.0 / (xMax - xMin)
    y = numpy.round((x - xMin) * scale_factor)
    return y.astype(numpy.uint8)

H1 = numpy.zeros((256, 256))
H1[126:133, 126:133] = 1/49

H2 = numpy.fft.fftshift(H1)
print(H2.shape)

pl.figure(figsize=(10, 6))
pl.subplot(2, 2, 1)
pl.imshow(stretch(X), cmap='gray')
pl.title('Zero Phase Impulse Resp', fontsize=18)
pl.axis('image')
pl.axis('off')

pl.subplot(2, 2, 2)
pl.imshow(stretch(H2), cmap='gray')
pl.title('Zero Phase Impulse Resp', fontsize=18)
pl.axis('image')
pl.axis('off')

ZPX = numpy.zeros((512, 512))
ZPX[:256, :256] = X

ZPH2 = numpy.zeros((512, 512))
ZPH2[:128, :128] = H2[:128, :128]
ZPH2[:128, 385:512] = H2[:128, 129:256]
ZPH2[385:512, :128] = H2[129:256, :128]
ZPH2[385:512, 385:512] = H2[129:256, 129:256]

pl.subplot(2, 2, 3)
pl.imshow(stretch(ZPH2), cmap='gray')
pl.title('Zero Padded zero-phase H', fontsize=18)
pl.axis('image')
pl.axis('off')

Y = numpy.fft.ifft2(numpy.fft.fft2(ZPX) * numpy.fft.fft2(ZPH2))
Y = stretch(Y[:256, :256])

pl.subplot(2, 2, 4)
pl.imshow(Y, cmap='gray')
pl.title('Final Filtered Image', fontsize=18)
pl.axis('image')
pl.axis('off')
pl.show()


