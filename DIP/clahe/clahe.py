import numpy as np
import cv2


def clip_histogram(hist, clip_limit):
    # calculate total number of excess pixels
    excess_mask = hist > clip_limit
    excess = hist[excess_mask]
    n_excess = excess.sum() - excess.size * clip_limit
    hist[excess_mask] = clip_limit

    # Second part: clip histogram and redistribute excess pixels in each bin
    bin_incr = n_excess // hist.size  # average binincrement
    upper = clip_limit - bin_incr  # Bins larger than upper set to cliplimit

    low_mask = hist < upper
    n_excess -= hist[low_mask].size * bin_incr
    hist[low_mask] += bin_incr

    mid_mask = np.logical_and(hist >= upper, hist < clip_limit)
    mid = hist[mid_mask]
    n_excess += mid.sum() - mid.size * clip_limit
    hist[mid_mask] = clip_limit

    while n_excess > 0:  # Redistribute remaining excess
        prev_n_excess = n_excess
        for index in range(hist.size):
            under_mask = hist < clip_limit
            step_size = max(1, np.count_nonzero(under_mask) // n_excess)
            under_mask = under_mask[index::step_size]
            hist[index::step_size][under_mask] += 1
            n_excess -= np.count_nonzero(under_mask)
            if n_excess <= 0:
                break
        if prev_n_excess == n_excess:
            break

    return hist

def conster(hist):
    C = int(hist.max()) + int(hist.min())//2
    top = C
    bottom = hist.min()
    middle = 0
    while top - bottom > 1:
        middle = (top + bottom) // 2
        excess_true = (hist - middle) > 0
        sum = (hist[excess_true] - middle)
        s = sum.sum()
        if s > (C - middle) * len(hist):
            top = middle
        else:
            bottom = middle
    top = middle

    excess_true = (hist - top) > 0
    sum = (hist[excess_true]-top)
    s = sum.sum()
    hist[excess_true] = top
    hist += s//len(hist)

    return hist


def histr(img):
    size = img.shape
    hist, t = np.histogram(img, range=(0, 256), bins=256)
    #hist = conster(hist)
    hist = clip_histogram(hist, (hist.max() + hist.min())//2).astype(float)
    hist = hist/hist.sum()
    s = hist.cumsum()
    s = np.floor(s*255)
    img = s[img.flatten()]
    return s, np.reshape(img, size)


def lineal(x, h_l, h_r,  x_minus, x_plus):
    a = (x_plus - x) / (x_plus - x_minus)
    b = (x - x_minus) / (x_plus - x_minus)
    m = a * h_l + b * h_r
    return int(m)


def lin_boardx(hist, maps, img, i, j, x0, x1, y0, y1, stepx, stepy):
    for k in range(x0, x1 + 1):
        for l in range(y0, y1 + 1):
            hist[k, l] = lineal(k, maps[x0//stepx][y1//stepy][img[k, l]], maps[x1//stepx][y1//stepy][img[k, l]], x0, x1)

    return hist


def lin_boardy(hist, maps, img, i, j, x0, x1, y0, y1, stepx, stepy):
    for k in range(x0, x1 + 1):
        for l in range(y0, y1 + 1):
            hist[k, l] = lineal(l, maps[x1//stepx][y0//stepy][img[k, l]], maps[x1//stepx][y1//stepy][img[k, l]], y0, y1)

    return hist


def ahe(img):

    h, w = img.shape

    stepx = h//8
    stepy = w//8

    lastx = h % 8
    lasty = w % 8

    centrx = stepx//2
    centry = stepy//2

    hist = np.zeros(img.shape)
    maps = []

    for i in range(0, 8):
        map_row = []
        for j in range(0, 8):
            x0 = i*stepx
            x1 = i*stepx+stepx
            y0 = j*stepy
            y1 = j*stepy+stepy
            if i < 7 and j < 7:
                s, hist[x0:x1, y0:y1] = histr(img[x0:x1, y0:y1])
                map_row.append(s)
            if i == 7 and j < 7:
                s, hist[x0:x1+lastx, y0:y1] = histr(img[x0:x1+lastx, y0:y1])
                map_row.append(s)
            if i < 7 and j == 7:
                s, hist[x0:x1, y0:y1+lasty] = histr(img[x0:x1, y0:y1+lasty])
                map_row.append(s)
            if i == 7 and j == 7:
                s, hist[x0:x1+lastx, y0:y1+lasty] = histr(img[x0:x1+lastx, y0:y1+lasty])
                map_row.append(s)

        maps.append(map_row)

    for i in range(1, 8):
        for j in range(1, 8):

            x0 = i * stepx - centrx
            x1 = i * stepx + stepx - centrx
            y0 = j * stepy - centry
            y1 = j * stepy + stepy - centry

            x_minus = x0
            x_plus = x1
            y_minus = y0
            y_plus = y1

            for k in range(x_minus, x_plus + 1):
                for l in range(y_minus, y_plus + 1):
                    c = (x_plus-k)/(x_plus - x_minus)
                    d = (k - x_minus)/(x_plus - x_minus)
                    a = (y_plus - l)/(y_plus-y_minus)
                    b = (l - y_minus)/(y_plus-y_minus)
                    f1 = a*maps[x_minus//stepx][y_minus//stepy][img[k, l]] + b*maps[x_minus//stepx][y_plus//stepy][img[k, l]]
                    f2 = a*maps[x_plus//stepx][y_minus//stepy][img[k, l]] + b*maps[x_plus//stepx][y_plus//stepy][img[k, l]]
                    m = c * f1 + d*f2
                    hist[k, l] = int(m)

    '''
    hist[0:centrx, 0:centry] += hist[centrx, centry]
    hist[0:centrx, 7*stepy + centry::] += hist[centrx, 7*stepy + centry]
    hist[7*stepx + centrx::, 0:centry] += hist[7*stepx + centrx, centry]
    hist[7*stepx + centrx::, 7*stepy + centry::] += hist[7*stepx + centrx, 7*stepy + centry]
    '''
    for i in range(0, 7):
        x0 = i * stepx + centrx
        x1 = i * stepx + centrx + stepx
        y0 = 0 * stepy
        y1 = 0 * stepy + centry
        hist = lin_boardx(hist, maps, img, i, 0, x0, x1, y0, y1, stepx, stepy)

    for i in range(0, 7):
        x0 = i * stepx + centrx
        x1 = i * stepx + centrx + stepx
        y0 = 7 * stepy + centry
        y1 = w - 1
        hist = lin_boardx(hist, maps, img, i, 7, x0, x1, y0, y1, stepx, stepy)

    for i in range(0, 7):
        y0 = i * stepy + centry
        y1 = i * stepy + centry + stepy
        x0 = 0 * stepx
        x1 = 0 * stepx + centrx
        hist = lin_boardy(hist, maps, img, 0, i, x0, x1, y0, y1, stepx, stepy)

    for i in range(0, 7):
        y0 = i * stepy + centry
        y1 = i * stepy + centry + stepy
        x0 = 7 * stepx + centrx
        x1 = h - 1
        hist = lin_boardy(hist, maps, img, 7, i, x0, x1, y0, y1, stepx, stepy)

    minxs = hist.min()
    maxs = hist.max()
    hist = (hist - minxs)/(maxs-minxs)
    hist *= 255

    return np.floor(hist)

if __name__ == "__main__":

    #файл
    path = 'image.png'
    img = cv2.imread(path)

    img_res = img

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(img.shape)

    im = ahe(img)
    print(im.shape)
    cv2.imwrite('res.png', im)






