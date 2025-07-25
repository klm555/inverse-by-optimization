#%% MP4

import cv2
import os
import glob

def png_to_mp4_with_opencv(image_dir, output_path, fps=30):
    """
    image_dir: PNG 이미지가 있는 디렉터리 경로
    output_path: 생성할 mp4 파일 경로
    fps: 초당 프레임 수(기본값: 30)
    """
    # image_dir/*.png 패턴으로 PNG 이미지 찾고, sorted()로 정렬
    rel_file_paths = sorted(glob.glob(os.path.join(image_dir, "ani.*.png")))

    # Convert to absolute path
    abs_file_paths = [os.path.abspath(fp) for fp in rel_file_paths]
    
    if not abs_file_paths:
        print("PNG 이미지를 찾지 못했습니다.")
        return

    # 첫 번째 이미지로 영상 크기 추출
    first_frame = cv2.imread(abs_file_paths[0])
    height, width, _ = first_frame.shape
    
    # VideoWriter 객체 생성
    # fourcc 코덱: 'mp4v' 또는 'XVID' 등
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 모든 이미지를 순회하며 동영상에 작성
    for path in abs_file_paths:
        img = cv2.imread(path)
        # 혹시 사이즈가 다를 수 있으니 필요시 resize 등을 적용
        out.write(img)
    
    out.release()
    print(f"동영상이 생성되었습니다: {output_path}")


if __name__ == "__main__":
    # 예시 사용
    image_dir = r'C:\Users\krist\jax-fem\data\inverse\dogbone_0.01_any_iter500\figures'
    output_file = os.path.join(image_dir, "output.mp4")
    png_to_mp4_with_opencv(image_dir, output_file, fps=14.3)