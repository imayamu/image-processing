import cv2
import numpy as np

def dct_transform(image, threshold):
    # 画像読み込み
    img = np.array(image, dtype=np.float32)
    img2 = np.array(image1, dtype=np.float32)
    
    
    # 画像の高さと幅を取得
    height, width = img.shape
    
    # 幅と高さを8の倍数に調整
    
    new_width = (width // 8) * 8
    new_height = (height // 8) * 8
    img = img[:new_height, :new_width]
    
    # DCT係数の総和と対応するi, jの値を格納するリストを作成
    dct_sums = []
    
    # 8x8ブロックごとにDCT変換を行う
    for m in range((height-8) // 8):
        for n in range((width-8) // 8):
            for s in range(8):
                for t in range(8):
                    # 8x8ブロックの切り出し
                    block = img[m*8 + s:m*8 + s + 8, n*8 + t:n*8 + t + 8]
                    
                    # DCT変換を行う
                    dct_block = cv2.dct(block)
                    
                    # 指定された19成分のDCT係数の総和を計算
                    dct_coefficients = [
                        (0, 7), (7, 0), (1, 7), (7, 1),
                        (2, 7), (7, 2), (3, 7), (7, 3),
                        (4, 7), (7, 4), (5, 7), (7, 5),
                        (6, 7), (7, 6), (7, 7), (0, 6),
                        (6, 0), (1, 6), (6, 1)
                    ]
                    dct_sum = sum(abs(dct_block[row, col]) for row, col in dct_coefficients if row < 8 and col < 8)
                    
                    # 総和が一定以上の場合、(i, j)成分を黒く書き換える
                    if dct_sum >= threshold:
                        img2[m*8 + s:m*8 + s + 8, n*8 + t:n*8 + t + 8] = 0
                        #img2[m*8 + s, n*8 + t] = 0
                        
                        
    
    return img2

# 画像ファイルのパス
image_path = 'ufo2.jpg'

# 画像読み込み
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
image1 = cv2.imread(image_path)


# 一定以上の総和の閾値
threshold = 100

# DCT変換を実行し、成分を黒く書き換える
modified_image = dct_transform(image, threshold)

# 画像表示
cv2.imwrite('DCT22ufo2.jpg',modified_image)