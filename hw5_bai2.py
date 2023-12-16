import cv2
import numpy
import matplotlib.pyplot as pl


def read_bin(file_path, size):
    with open(file_path, 'rb') as file:
        data = numpy.fromfile(file, dtype=numpy.uint8, count=size*size)
        return numpy.reshape(data, (size, size))


X = read_bin('../image/girl2bin.sec', 256)
X1 = read_bin('../image/girl2Noise32bin.sec', 256)
X2 = read_bin('../image/girl2Noise32Hibin.sec', 256)

E1 = numpy.sum((X.astype("float") - X1.astype("float"))**2)
E1 /= float(X.shape[0] * X1.shape[1])

E2 = numpy.sum((X.astype("float") - X2.astype(float))**2)
E2 /= float(X.shape[0] * X2.shape[1])

print("Mean Squared Error between girl2 and girl2Noise32 = ", E1)
print("Mean Squared Error between girl2 and girl2Noise32Hi = ", E2)

pl.figure(figsize=(12, 6))
pl.subplot(1, 3, 1)
pl.imshow(X, cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('Original Image', fontsize=18)

pl.subplot(1, 3, 2)
pl.imshow(X1, cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('girl2Noise32 Image', fontsize=18)

pl.subplot(1, 3, 3)
pl.imshow(X2, cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('girl2Noise32Hi Image', fontsize=18)

pl.show()



def stretch2(img):
    img_min = numpy.min(img)
    img_max = numpy.max(img)
    stretched_img = 255 * (img - img_min) / (img_max - img_min)
    return stretched_img.astype(numpy.uint8)


U_cutoff = 64
[U, V] = numpy.meshgrid(numpy.arange(-128, 128), numpy.arange(-128, 128))

HLtildeCenter = numpy.double(numpy.sqrt(U**2 + V**2) <= U_cutoff)
HLtilde = numpy.fft.fftshift(HLtildeCenter)

Z = numpy.fft.ifft2(numpy.fft.fft2(X) * HLtilde)
Z1 = numpy.fft.ifft2(numpy.fft.fft2(X1) * HLtilde)
Z2 = numpy.fft.ifft2(numpy.fft.fft2(X2) * HLtilde)

pl.figure(figsize=(12, 6))
pl.subplot(1, 3, 1)
pl.imshow(stretch2(Z), cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('Original Image', fontsize=18)

pl.subplot(1, 3, 2)
pl.imshow(stretch2(Z1), cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('girl2Noise32 Image', fontsize=18)

pl.subplot(1, 3, 3)
pl.imshow(stretch2(Z2), cmap='gray', vmin=0, vmax=255)
pl.axis('image')
pl.axis('off')
pl.title('girl2Noise32Hi Image', fontsize=18)

pl.show()

EZ = numpy.sum(numpy.abs(X.astype("complex") - Z.astype("complex")) ** 2)
EZ /= float(X.shape[0] * Z.shape[1])

EZ1 = numpy.sum(numpy.abs(X.astype("complex") - Z1.astype("complex")) ** 2)
EZ1 /= float(X.shape[0] * Z1.shape[1])

EZ2 = numpy.sum(numpy.abs(X.astype("complex") - Z2.astype("complex")) ** 2)
EZ2 /= float(X.shape[0] * Z2.shape[1])

ISNR1 = 10*numpy.log10(E1/EZ1)
ISNR2 = 10*numpy.log10(E2/EZ2)

print('MSE of Z', EZ)
print('MSE of Z1', EZ1)
print('MSE of Z2', EZ2)
print('ISNR of girl2Noise32Hibin = ', ISNR2)
print('ISNR of girl2Noise32bin = ', ISNR1)



def GaussianLPF(U_cutoff_H, X, X1, X2):
    SigmaH = 0.19 * 256 / U_cutoff_H
    U, V = numpy.meshgrid(numpy.arange(-128, 128), numpy.arange(-128, 128))
    HtildeCenter = numpy.exp((-2 * numpy.pi ** 2 * SigmaH ** 2) / (256 ** 2) * (U ** 2 + V ** 2))
    Htilde = numpy.fft.fftshift(HtildeCenter)
    H = numpy.fft.ifft2(Htilde)
    H2 = numpy.fft.fftshift(H)
    ZPH2 = numpy.zeros((512, 512))
    ZPH2[:256, :256] = H2

    ZPX = numpy.zeros((512, 512))
    ZPX[:256, :256] = X
    yy = numpy.fft.ifft2(numpy.fft.fft2(ZPX) * numpy.fft.fft2(ZPH2))
    T = yy[128:384, 128:384]

    ZPX[:256, :256] = X1
    yy = numpy.fft.ifft2(numpy.fft.fft2(ZPX) * numpy.fft.fft2(ZPH2))
    T1 = yy[128:384, 128:384]

    ZPX[:256, :256] = X2
    yy = numpy.fft.ifft2(numpy.fft.fft2(ZPX) * numpy.fft.fft2(ZPH2))
    T2 = yy[128:384, 128:384]

    ET = numpy.sum(numpy.abs(X.astype("complex") - T.astype("complex")) ** 2)
    ET /= float(X.shape[0] * T.shape[1])

    ET1 = numpy.sum(numpy.abs(X.astype("complex") - T1.astype("complex")) ** 2)
    ET1 /= float(X.shape[0] * T1.shape[1])

    ET2 = numpy.sum(numpy.abs(X.astype("complex") - T.astype("complex")) ** 2)
    ET2 /= float(X.shape[0] * T2.shape[1])

    ISNR3 = 10 * numpy.log10(E1 / ET1)
    ISNR4 = 10 * numpy.log10(E2 / ET2)

    print('MSE of girl2bin', ET)
    print('MSE of girl2Noise32bin', ET1)
    print('MSE of girl2Noise32Hibin', ET2)
    print('ISNR of girl2Noise32Hibin = ', ISNR3)
    print('ISNR of girl2Noise32bin = ', ISNR4)

    pl.figure(figsize=(12, 6))
    pl.subplot(1, 3, 1)
    pl.imshow(stretch2(T), cmap='gray', vmin=0, vmax=255)
    pl.axis('image')
    pl.axis('off')
    pl.title('Original Image', fontsize=18)

    pl.subplot(1, 3, 2)
    pl.imshow(stretch2(T1), cmap='gray', vmin=0, vmax=255)
    pl.axis('image')
    pl.axis('off')
    pl.title('girl2Noise32 Image', fontsize=18)

    pl.subplot(1, 3, 3)
    pl.imshow(stretch2(T2), cmap='gray', vmin=0, vmax=255)
    pl.axis('image')
    pl.axis('off')
    pl.title('girl2Noise32Hi Image', fontsize=18)

    pl.show()


print('------------------Câu c---------------------')
GaussianLPF(64, X, X1, X2)
print('------------------Câu d---------------------')
GaussianLPF(77.5, X, X1, X2)














