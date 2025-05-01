from PIL import Image, ImageDraw, ImageFont
import os
import math
import random

from utils.common_chars import common_chars_3500, common_symbol

# 设置保存目录

# 生成图片的数量
num_images = 100
image_size = 256  # 统一图片大小

# 设置文本长度范围
min_length = 7  # 最小字符长度
max_length = 10 # 最大字符长度

output_dir = f'images_{max_length}'
os.makedirs(output_dir, exist_ok=True)


# 选择支持中文的字体（请更换为你系统中的字体）
font_path = "/usr/share/fonts/windows/simsun.ttc"  # Windows 系统字体


# font_path = "/usr/share/fonts/opentype/noto/NotoSansSC-Bold.otf"  # Linux
# font_path = "/System/Library/Fonts/Supplemental/Songti.ttc"  # macOS


def random_chinese_char():
    # 随机生成一个汉字的Unicode编码，范围是 \u4e00 到 \u9fff
    # unicode_code = random.randint(0x4e00, 0x62ff)
    # return chr(unicode_code)
    if random.random() < 0.8:
        # 随机生成一个汉字
        return random.choice(common_chars_3500)
    else:
        # 随机生成一个符号
        return random.choice(common_symbol)


for i in range(1, num_images + 1):
    # 随机生成文本长度
    num_chars = random.randint(min_length, max_length)

    # 生成随机中文文本
    text = ''.join(random_chinese_char() for _ in range(num_chars))

    #print(f"Generated text: {text}")
    #print(f"num_char:{num_chars}")

    # 计算矩阵排布大小（√n）
    grid_size = math.ceil(math.sqrt(num_chars)) if num_chars > 0 else 1  # 每行字符数
    #print(f"grid_size:{grid_size}")
    font_size = int(image_size / grid_size)  # 计算字体大小
    #print(f"font_size:{font_size}")
    font = ImageFont.truetype(font_path, font_size)

    # 创建灰度图像
    img = Image.new('L', (image_size, image_size), color=255)
    draw = ImageDraw.Draw(img)

    # 计算每行的文本，并按行排列
    lines = [text[i:i + grid_size] for i in range(0, num_chars, grid_size)]

    # 获取每行的最大宽度和总高度
    line_heights = []
    max_width = 0
    for line in lines:
        text_bbox = draw.textbbox((0, 0), line, font=font)  # 使用 textbbox()
        line_width = text_bbox[2] - text_bbox[0]
        line_height = text_bbox[3] - text_bbox[1]
        line_heights.append(line_height)
        max_width = max(max_width, line_width)

    total_height = sum(line_heights)  # 所有行的总高度

    # 计算起始坐标，使文本居中
    y_offset = (image_size - total_height) // 2  # Y方向居中
    x_offset = (image_size - max_width) // 2  # X方向居中

    # 在图片上绘制文本
    for line, line_height in zip(lines, line_heights):
        draw.text((x_offset, y_offset), line, font=font, fill=0)
        y_offset += line_height  # 更新Y坐标，绘制下一行

    # 保存图片，文件名为编号.png
    img.save(f'{output_dir}/{i}.png')

print(f"生成了 {num_images} 张图片，保存在 {output_dir} 目录下。")
