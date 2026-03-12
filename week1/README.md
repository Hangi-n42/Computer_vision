# **Week 1**

## 1. 이미지 불러오기 및 그레이스케일 변환 
```python
import cv2 as cv
import numpy as np
import os

# soccer.jpg 이미지 로드
img = cv.imread('soccer.jpg')
if img is None:
    print("오류: soccer.jpg 파일을 찾을 수 없습니다.")
    exit()
    
# BGR 이미지를 그레이스케일로 변환
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# 그레이스케일을 3채널로 변환하여 원본과 같은 형식으로 만들기
gray_3channel = np.dstack([gray, gray, gray])

# 이미지들을 리스트에 저장
frames = [img, gray_3channel]

# 이미지들을 나란히 이어붙히기
if len(frames) > 0:
    imgs = frames[0]
    for i in range(1, len(frames)):
        imgs = np.hstack((imgs, frames[i]))

# 결과 이미지 표시
cv.namedWindow('Original and Grayscale Image', cv.WINDOW_NORMAL)
cv.resizeWindow('Original and Grayscale Image', imgs.shape[1], imgs.shape[0])
cv.imshow('Original and Grayscale Image', imgs)

cv.waitKey()
cv.destroyAllWindows()

print(type(img))
print(img.shape)
print("Grayscale shape:", gray.shape)
```
1. cv.imread()를 사용해 soccer.jpg 이미지를 로드하고, 파일이 없을 경우 None 검사를 통해 오류 메시지를 출력한 뒤 프로그램을 종료한다.
2. cv.cvtColor(img, cv.COLOR_BGR2GRAY)로 원본 BGR 이미지를 그레이스케일로 변환한다.
3. 그레이스케일 이미지는 1채널이므로, 원본과 같은 형태로 나란히 붙이기 위해 np.dstack([gray, gray, gray])로 3채널 이미지로 변환한다.
4. 원본 이미지와 변환된 그레이 이미지를 리스트(frames)에 저장한 뒤, np.hstack()을 사용해 두 이미지를 가로 방향으로 연결한다.
5. cv.namedWindow()와 cv.resizeWindow()로 결과 창 크기를 이미지 크기에 맞춰 설정하고, cv.imshow()로 결합 결과를 화면에 출력한다.
6. cv.waitKey()로 키 입력을 기다린 후 cv.destroyAllWindows()를 호출해 모든 창을 종료한다.

<img width="1919" height="1022" alt="Image" src="https://github.com/user-attachments/assets/c7c6b4c0-ecfc-4187-9224-cabf58d23344" />


  

## 2. 페인팅 붓 크기 조절 기능 추가
```python
import cv2 as cv
import numpy as np

drawing = False  # 드래그 중인지 여부
brush_size = 5  # 초기 붓 크기
brush_color = (255, 0, 0)  # 초기 색상 (파란색)

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수"""
    global drawing, brush_color, image
    
    if event == cv.EVENT_LBUTTONDOWN:
        # 좌클릭 - 파란색
        drawing = True
        brush_color = (255, 0, 0)  # BGR: 파란색
        cv.circle(image, (x, y), brush_size, brush_color, -1)
        
    elif event == cv.EVENT_RBUTTONDOWN:
        # 우클릭 - 빨간색
        drawing = True
        brush_color = (0, 0, 255)  # BGR: 빨간색
        cv.circle(image, (x, y), brush_size, brush_color, -1)
        
    elif event == cv.EVENT_MOUSEMOVE:
        # 드래그로 연속 그리기
        if drawing:
            cv.circle(image, (x, y), brush_size, brush_color, -1)
            
    elif event == cv.EVENT_LBUTTONUP or event == cv.EVENT_RBUTTONUP:
        # 마우스 버튼을 떼면 그리기 중단
        drawing = False

# soccer.jpg 이미지 로드
image = cv.imread('soccer.jpg')
if image is None:
    print("오류: soccer.jpg 파일을 찾을 수 없습니다.")
    exit()

# 윈도우 생성 및 마우스 콜백 설정
window_name = 'Painting'
cv.namedWindow(window_name)
cv.setMouseCallback(window_name, mouse_callback)

print("=== 페인팅 프로그램 ===")
print("좌클릭: 파란색으로 그리기")
print("우클릭: 빨간색으로 그리기")
print("드래그: 연속으로 그리기")
print("+: 붓 크기 증가")
print("-: 붓 크기 감소")
print("q: 종료")    
print(f"현재 붓 크기: {brush_size}")

while True:
    # 현재 붓 크기를 화면에 표시
    display_image = image.copy()
    cv.putText(display_image, f'Brush Size: {brush_size}', (10, 30), 
               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv.imshow(window_name, display_image)
    
    # 키 입력 대기 (1ms)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # q 키를 누르면 종료
        print("프로그램을 종료합니다.")
        break
    elif key == ord('+') or key == ord('='):
        # + 키를 누르면 붓 크기 증가 (최대 15)
        if brush_size < 15:
            brush_size += 1
            print(f"붓 크기 증가: {brush_size}")
    elif key == ord('-') or key == ord('_'):
        # - 키를 누르면 붓 크기 감소 (최소 1)
        if brush_size > 1:
            brush_size -= 1
            print(f"붓 크기 감소: {brush_size}")

cv.destroyAllWindows()
```
1. cv.imread()로 배경 이미지(soccer.jpg)를 로드한 뒤, cv.namedWindow()와 cv.setMouseCallback()으로 마우스 입력을 받는 페인팅 창을 구성한다.
2. mouse_callback() 함수에서 좌클릭은 파란색, 우클릭은 빨간색으로 설정하고, 클릭 시 cv.circle()을 그려 바로 페인팅이 시작되도록 한다.
3. 마우스 이동 이벤트에서 drawing 상태가 True인 동안 cv.circle()을 반복 호출해 드래그하는 경로를 따라 연속적으로 그리도록 구현한다.
4. 마우스 버튼을 떼는 이벤트(EVENT_LBUTTONUP, EVENT_RBUTTONUP)에서 drawing을 False로 바꿔 그리기를 중단한다.
5. 메인 루프에서는 현재 이미지를 복사해 cv.putText()로 붓 크기를 표시하고, cv.imshow()로 실시간 화면을 갱신한다.
6. cv.waitKey(1)로 키 입력을 확인해 '+'/'=' 입력 시 붓 크기를 증가시키고, '-'/'_' 입력 시 붓 크기를 감소시키며, 'q' 입력 시 루프를 종료한다.
7. 프로그램 종료 시 cv.destroyAllWindows()를 호출해 창을 정리한다.

<img width="1790" height="1027" alt="Image" src="https://github.com/user-attachments/assets/5b887fd5-2b00-4571-a9c2-26310590e4f9" />

  

## 3. 마우스로 영역 선택 및 ROI(관심영역) 추출
```python
import cv2 as cv
import numpy as np

drawing = False  # 드래그 중인지 여부
start_point = None  # 시작점
end_point = None  # 끝점
roi_selected = False  # ROI 선택 완료 여부
original_image = None  # 원본 이미지
display_image = None  # 화면에 표시할 이미지

def mouse_callback(event, x, y, flags, param):
    """마우스 이벤트 콜백 함수"""
    global drawing, start_point, end_point, roi_selected, display_image
    
    if event == cv.EVENT_LBUTTONDOWN:
        # 좌클릭 - 시작점 설정
        drawing = True
        roi_selected = False
        start_point = (x, y)
        end_point = (x, y)
        
    elif event == cv.EVENT_MOUSEMOVE:
        
        # 드래그 중 - 사각형 그리기
        if drawing:
            end_point = (x, y)
            # 원본 이미지 복사하여 사각형 그리기
            display_image = original_image.copy()
            cv.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
            
    elif event == cv.EVENT_LBUTTONUP:
        # 마우스 버튼을 떼면 ROI 선택 완료
        drawing = False
        roi_selected = True
        end_point = (x, y)
        # 최종 사각형 그리기
        display_image = original_image.copy()
        cv.rectangle(display_image, start_point, end_point, (0, 255, 0), 2)
        
        # ROI 영역 추출 및 표시
        extract_and_show_roi()

def extract_and_show_roi():
    """ROI 영역을 추출하고 별도 창에 표시"""
    global start_point, end_point, original_image
    
    if start_point is None or end_point is None:
        return
    
    # 좌표 정렬 (왼쪽 위, 오른쪽 아래)
    x1 = min(start_point[0], end_point[0])
    y1 = min(start_point[1], end_point[1])
    x2 = max(start_point[0], end_point[0])
    y2 = max(start_point[1], end_point[1])
    
    # 유효한 영역인지 확인
    if x2 > x1 and y2 > y1:
        # numpy 슬라이싱으로 ROI 추출
        roi = original_image[y1:y2, x1:x2]
        
        if roi.size > 0:
            cv.imshow('ROI', roi)
            print(f"ROI 선택 완료: ({x1}, {y1}) ~ ({x2}, {y2})")

def reset_selection():
    """영역 선택 리셋"""
    global start_point, end_point, roi_selected, display_image
    start_point = None
    end_point = None
    roi_selected = False
    display_image = original_image.copy()
    cv.destroyWindow('ROI')
    print("영역 선택이 리셋되었습니다.")

def save_roi():
    """선택한 ROI를 파일로 저장"""
    global start_point, end_point, original_image, roi_selected
    
    if not roi_selected or start_point is None or end_point is None:
        print("저장할 ROI가 선택되지 않았습니다.")
        return
    
    # 좌표 정렬
    x1 = min(start_point[0], end_point[0])
    y1 = min(start_point[1], end_point[1])
    x2 = max(start_point[0], end_point[0])
    y2 = max(start_point[1], end_point[1])
    
    if x2 > x1 and y2 > y1:
        # ROI 추출
        roi = original_image[y1:y2, x1:x2]
        
        if roi.size > 0:
            # 파일 저장
            filename = 'roi_selected.jpg'
            cv.imwrite(filename, roi)
            print(f"ROI가 '{filename}'로 저장되었습니다.")
        else:
            print("저장할 영역이 비어있습니다.")
    else:
        print("유효하지 않은 영역입니다.")

# 이미지 로드
original_image = cv.imread('soccer.jpg')
if original_image is None:
    print("오류: soccer.jpg 파일을 찾을 수 없습니다.")
    exit()

display_image = original_image.copy()

# 윈도우 생성 및 마우스 콜백 설정
window_name = 'Image ROI Selection'
cv.namedWindow(window_name)
cv.setMouseCallback(window_name, mouse_callback)

print("=== ROI 선택 프로그램 ===")
print("마우스 클릭 & 드래그: ROI 영역 선택")
print("r: 영역 선택 리셋")
print("s: 선택한 영역을 이미지 파일로 저장")
print("q: 종료")

while True:
    cv.imshow(window_name, display_image)
    
    # 키 입력 대기 (1ms)
    key = cv.waitKey(1) & 0xFF
    
    if key == ord('q'):
        # q 키를 누르면 종료
        print("프로그램을 종료합니다.")
        break
    elif key == ord('r'):
        # r 키를 누르면 영역 선택 리셋
        reset_selection()
    elif key == ord('s'):
        # s 키를 누르면 선택한 영역 저장
        save_roi()

cv.destroyAllWindows()
```
1. cv.imread()로 원본 이미지(soccer.jpg)를 로드하고, display_image를 복사본으로 준비한 뒤 ROI 선택용 창을 생성하고 마우스 콜백을 등록한다.
2. mouse_callback()에서 좌클릭으로 시작점(start_point)을 잡고, 드래그 중에는 원본 복사본 위에 cv.rectangle()로 현재 선택 영역을 실시간 표시한다.
3. 마우스 버튼을 놓으면 선택이 완료되며, extract_and_show_roi()를 호출해 시작점과 끝점을 기준으로 ROI를 추출한다.
4. ROI 추출 시에는 min/max를 사용해 좌표를 정렬한 후, original_image[y1:y2, x1:x2] 형태의 NumPy 슬라이싱으로 유효 영역만 잘라낸다.
5. 추출된 영역이 비어 있지 않으면 cv.imshow('ROI', roi)로 별도 창에 표시하고, 선택 좌표를 출력해 사용자가 확인할 수 있게 한다.
6. 키 입력 처리에서는 'r'로 선택 상태를 초기화(reset_selection), 's'로 현재 ROI를 cv.imwrite()로 파일(roi_selected.jpg)로 저장, 'q'로 프로그램을 종료한다.
7. 마지막에 cv.destroyAllWindows()를 호출해 메인 창과 ROI 창을 모두 닫는다.

<img width="1791" height="1017" alt="Image" src="https://github.com/user-attachments/assets/65bedabf-44a6-4844-83db-b704c2fa6f06" />
