import cv2


img=cv2.imread(r'E:\\dataset1\image_8.png')
cv2.imshow('img',img)

# key = cv2.waitKey(0)
# if key == ord('q'):  # 如果按下 'q' 键，退出
#     cv2.destroyAllWindows()
print(type(img))
print(img.shape)
print(img)
