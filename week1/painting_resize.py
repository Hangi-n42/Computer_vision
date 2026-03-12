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
