from ultralytics import YOLO
import cv2
import numpy as np
import os
import pygame
import win32gui
import win32con
import win32api
import win32ui
import tkinter as tk
from tkinter import ttk, messagebox
import time
import threading
import ctypes
from ctypes import wintypes
import gc
import psutil

class DesktopMonitor:
    def __init__(self):
        # 启用垃圾回收
        gc.enable()
        gc.set_threshold(700, 10, 5)  # 调整垃圾回收阈值，减少GC频率
        
        # 获取当前进程
        self.process = psutil.Process()
        
        # 设置更合理的内存阈值
        self.memory_warning_threshold = 1000 * 1024 * 1024  # 1000MB
        self.memory_limit = 1.5 * 1024 * 1024 * 1024  # 1.5GB
        
        # 使用指定的模型路径，并设置设备
        model_path = os.path.join(os.path.dirname(__file__), 'weights', 'yolo11n.pt')
        self.model = YOLO(model_path)
        
        # 初始化pygame用于播放音乐
        pygame.mixer.init()
        self.alert_sound = os.path.join(os.path.dirname(__file__), 'sounds', 'alert.mp3')
        
        # 设置保存路径
        self.save_dir = os.path.join(os.path.dirname(__file__), 'detections')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        # 状态标志
        self.is_playing = False
        self.target_window = None
        self.running = True
        self.monitoring = False
        
        # 检测和显示参数
        self.conf_threshold = 0.65  # 稍微降低置信度阈值
        self.fps_limit = 15  # 提高帧率
        self.frame_interval = 1.0 / self.fps_limit
        self.last_frame_time = 0
        self.scale_factor = 0.6  # 提高缩放比例
        self.class_colors = {}
        
        # 添加缓存机制
        self.frame_cache = []
        self.max_cache_frames = 5  # 增加缓存帧数
        self.last_detection_time = 0
        self.detection_interval = 0.1  # 减少检测间隔
        
        # 内存监控
        self.last_memory_check = 0
        self.memory_check_interval = 30  # 降低内存检查频率
        
        # 初始化Windows API
        self.user32 = ctypes.WinDLL('user32')
        self.user32.PrintWindow.argtypes = [wintypes.HWND, wintypes.HDC, wintypes.UINT]
        self.user32.PrintWindow.restype = wintypes.BOOL
        
        # 添加截图控制参数
        self.last_save_time = 0
        self.min_save_interval = 3.0  # 最小保存间隔(秒)
        self.last_detection_boxes = None
        self.detection_threshold = 0.2  # 检测框变化阈值
        self.continuous_detection_count = 0
        self.min_continuous_detections = 3  # 连续检测到才保存
        self.max_daily_images = 1000  # 每日最大图片数
        self.daily_image_count = 0
        self.last_date = time.strftime("%Y%m%d")
        
        # 创建日期子文件夹
        self.current_save_dir = os.path.join(self.save_dir, time.strftime("%Y%m%d"))
        if not os.path.exists(self.current_save_dir):
            os.makedirs(self.current_save_dir)
        
        # 创建主窗口
        self.create_main_window()

    def limit_memory(self, maxsize):
        """Windows系统下的内存管理"""
        try:
            # 设置工作集限制
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(
                kernel32.GetCurrentProcess(),
                -1,
                maxsize
            )
        except Exception as e:
            print(f"设置内存限制失败: {e}")

    def monitor_memory_usage(self):
        """监控内存使用情况"""
        try:
            # 获取当前进程的内存信息
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            print(f"当前内存使用: {memory_mb:.2f} MB")
            
            # 如果内存使用超过警告阈值，执行内存优化
            if memory_info.rss > self.memory_warning_threshold:
                print("内存使用过高，执行优化...")
                self.optimize_memory()
            
            # 如果内存使用超过限制，强制清理
            if memory_info.rss > self.memory_limit:
                print("内存使用超过限制，执行强制清理...")
                self.force_cleanup()
                
        except Exception as e:
            print(f"内存监控失败: {e}")

    def optimize_memory(self):
        """内存优化函数"""
        try:
            # 强制执行垃圾回收
            gc.collect()
            
            # 清理图像缓存
            self.cleanup_cache()
            
            # 清理OpenCV缓存
            cv2.destroyAllWindows()
            
            # 压缩进程工作集
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            
        except Exception as e:
            print(f"内存优化失败: {e}")

    def cleanup_cache(self):
        """清理缓存"""
        if hasattr(self, 'frame_cache'):
            # 只保留最后几帧
            self.frame_cache = self.frame_cache[-5:] if self.frame_cache else []
            
        # 确保删除任何未使用的大对象
        for attr in list(self.__dict__.keys()):
            if attr.startswith('_') and attr not in ['_saveBitMap', '_saveDC', '_mfcDC', '_hwndDC']:
                delattr(self, attr)

    def create_main_window(self):
        """创建主控制窗口"""
        self.root = tk.Tk()
        self.root.title("监控控制器")
        self.root.geometry("600x450")
        
        # 创建主框架
        frame = ttk.Frame(self.root)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # 创建树形视图
        columns = ("序号", "窗口标题", "句柄")
        self.tree = ttk.Treeview(frame, columns=columns, show="headings")
        
        # 设置列标题和宽度
        for col in columns:
            self.tree.heading(col, text=col)
        self.tree.column("序号", width=50)
        self.tree.column("窗口标题", width=400)
        self.tree.column("句柄", width=100)
        
        # 添加滚动条
        scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # 布局控件
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 按钮框架
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # 添加按钮
        self.refresh_btn = ttk.Button(button_frame, text="刷新列表", command=self.refresh_window_list)
        self.refresh_btn.pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(button_frame, text="开始监控", command=self.start_monitoring)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="停止监控", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        self.quit_btn = ttk.Button(button_frame, text="退出程序", command=self.quit_program)
        self.quit_btn.pack(side=tk.LEFT, padx=5)
        
        # 状态标签
        self.status_label = ttk.Label(self.root, text="就绪")
        self.status_label.pack(pady=5)
        
        # 初始加载窗口列表
        self.refresh_window_list()
        
        # 绑定双击事件
        self.tree.bind("<Double-1>", lambda e: self.start_monitoring())
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.quit_program)

    def cleanup_resources(self):
        """清理资源"""
        try:
            if hasattr(self, '_saveBitMap'):
                try:
                    self._saveBitMap.DeleteObject()
                except:
                    pass
                delattr(self, '_saveBitMap')
                
            if hasattr(self, '_saveDC'):
                try:
                    self._saveDC.DeleteDC()
                except:
                    pass
                delattr(self, '_saveDC')
                
            if hasattr(self, '_mfcDC'):
                try:
                    self._mfcDC.DeleteDC()
                except:
                    pass
                delattr(self, '_mfcDC')
                
            if hasattr(self, '_hwndDC') and self.target_window:
                try:
                    win32gui.ReleaseDC(self.target_window[0], self._hwndDC)
                except:
                    pass
                delattr(self, '_hwndDC')
            
            # 清理缓存
            self.cleanup_cache()
            
            # 强制垃圾回收
            gc.collect()
            
        except Exception as e:
            print(f"清理资源时发生错误: {e}")

    def quit_program(self):
        """退出程序"""
        # 先停止监控线程
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        # 清理资源
        self.cleanup_resources()
        
        # 设置运行标志为False
        self.running = False
        
        # 安全地销毁窗口
        if self.root.winfo_exists():
            try:
                self.root.quit()
                self.root.destroy()
            except:
                pass

    def list_windows(self):
        """列出所有可见的窗口"""
        windows = []
        def enum_windows_callback(hwnd, windows):
            if win32gui.IsWindowVisible(hwnd):
                window_text = win32gui.GetWindowText(hwnd)
                if window_text:
                    windows.append((hwnd, window_text))
            return True
        
        win32gui.EnumWindows(enum_windows_callback, windows)
        return windows

    def refresh_window_list(self):
        """刷新窗口列表"""
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        windows = self.list_windows()
        for i, (hwnd, title) in enumerate(windows):
            self.tree.insert("", tk.END, values=(i, title, hwnd))
    def start_monitoring(self):
        """开始监控选中的窗口"""
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("警告", "请选择要监控的窗口")
            return
        
        if self.monitoring:
            return
            
        item = self.tree.item(selection[0])
        _, title, hwnd = item['values']
        self.target_window = (hwnd, title)
        
        # 更新按钮状态
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"正在监控: {title}")
        
        # 开始监控线程
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1.0)
        
        # 只有在程序仍在运行时才更新按钮状态
        if self.running and self.root.winfo_exists():
            try:
                self.start_btn.config(state=tk.NORMAL)
                self.stop_btn.config(state=tk.DISABLED)
                self.status_label.config(text="就绪")
            except:
                pass
        
        # 清理���源
        self.cleanup_resources()
        
        try:
            cv2.destroyWindow("Detection Results")
        except:
            pass

    def check_detection_change(self, current_boxes):
        """检查检测框是否发生显著变化"""
        if self.last_detection_boxes is None:
            return True
            
        try:
            # 如果检测到的人数变化
            if len(current_boxes) != len(self.last_detection_boxes):
                return True
                
            # 计算检测框的变化程度
            total_change = 0
            for curr_box, last_box in zip(current_boxes, self.last_detection_boxes):
                curr_coords = curr_box.xyxy[0].cpu().numpy()
                last_coords = last_box.xyxy[0].cpu().numpy()
                
                # 计算框的中心点变化
                curr_center = [(curr_coords[0] + curr_coords[2])/2, (curr_coords[1] + curr_coords[3])/2]
                last_center = [(last_coords[0] + last_coords[2])/2, (last_coords[1] + last_coords[3])/2]
                
                # 计算中心点移动距离
                distance = np.sqrt((curr_center[0] - last_center[0])**2 + (curr_center[1] - last_center[1])**2)
                
                # 计算框的对角线长度作为参考
                diagonal = np.sqrt((curr_coords[2] - curr_coords[0])**2 + (curr_coords[3] - curr_coords[1])**2)
                
                # 如果移动距离超过对角线长度的阈值，认为发生了显著变化
                if distance > diagonal * self.detection_threshold:
                    return True
                    
            return False
            
        except Exception as e:
            print(f"检查检测变化时出错: {e}")
            return True

    def should_save_image(self, high_conf_boxes):
        """决定是否需要保存图片"""
        current_time = time.time()
        current_date = time.strftime("%Y%m%d")
        
        try:
            # 检查是否是新的一天
            if current_date != self.last_date:
                self.daily_image_count = 0
                self.last_date = current_date
                # 创建新的日期文件夹
                self.current_save_dir = os.path.join(self.save_dir, current_date)
                if not os.path.exists(self.current_save_dir):
                    os.makedirs(self.current_save_dir)
            
            # 检查是否超过每日最大图片数
            if self.daily_image_count >= self.max_daily_images:
                return False
            
            # 检查时间间隔
            if current_time - self.last_save_time < self.min_save_interval:
                return False
            
            # 检查检测框是否发生显著变化
            if not self.check_detection_change(high_conf_boxes):
                self.continuous_detection_count = 0
                return False
            
            # 增加连续检测计数
            self.continuous_detection_count += 1
            
            # 只有连续检测到才保存
            if self.continuous_detection_count >= self.min_continuous_detections:
                self.last_save_time = current_time
                self.daily_image_count += 1
                self.last_detection_boxes = high_conf_boxes
                return True
            
            return False
            
        except Exception as e:
            print(f"检查是否保存图片时出错: {e}")
            return False

    def monitor_loop(self):
        """优化的监控循环"""
        try:
            cv2.namedWindow("Detection Results", cv2.WINDOW_NORMAL)
            
            while self.monitoring and self.running:
                current_time = time.time()
                
                # 检查内存使用
                if current_time - self.last_memory_check >= self.memory_check_interval:
                    self.monitor_memory_usage()
                    self.last_memory_check = current_time
                
                # 控制帧率
                if current_time - self.last_frame_time < self.frame_interval:
                    time.sleep(0.001)
                    continue
                    
                self.last_frame_time = current_time
                
                # 捕获画面
                frame = self.capture_window()
                if frame is None:
                    time.sleep(0.001)
                    continue
                
                try:
                    # 缩放图像
                    small_frame = cv2.resize(frame, None, fx=self.scale_factor, fy=self.scale_factor, 
                                           interpolation=cv2.INTER_AREA)
                    
                    # 创建显示帧的副本
                    display_frame = frame.copy()
                    
                    # 检测间隔控制
                    should_detect = (current_time - self.last_detection_time) >= self.detection_interval
                    
                    if should_detect:
                        self.last_detection_time = current_time
                        
                        # 目标检测
                        results = self.model(small_frame, conf=self.conf_threshold)
                        
                        detected = False
                        for result in results:
                            boxes = result.boxes
                            person_boxes = boxes[boxes.cls == 0]
                            if len(person_boxes) > 0:
                                high_conf_boxes = [box for box in person_boxes if float(box.conf[0]) >= self.conf_threshold]
                                if high_conf_boxes:
                                    detected = True
                                    
                                    # 在显示帧上绘制检测结果
                                    for box in high_conf_boxes:
                                        # 调整检测框的坐标以匹配原始图像大小
                                        box_coords = box.xyxy[0].cpu().numpy()
                                        box_coords = box_coords / self.scale_factor
                                        x1, y1, x2, y2 = map(int, box_coords)
                                        
                                        # 绘制边框
                                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                        
                                        # 绘制置信度
                                        conf = float(box.conf[0])
                                        label = f"Person {conf:.2f}"
                                        cv2.putText(display_frame, label, (x1, y1-10), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                    
                                    # 判断是否需要保存图片
                                    if self.should_save_image(high_conf_boxes):
                                        # 保存检测结果
                                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                                        save_path = os.path.join(self.current_save_dir, f"detection_{timestamp}.jpg")
                                        cv2.imwrite(save_path, display_frame)
                                    
                                    # 播放警报
                                    self.play_alert()
                        
                        if not detected:
                            self.stop_alert()
                            self.continuous_detection_count = 0
                        
                        # 更新帧缓存
                        if len(self.frame_cache) >= self.max_cache_frames:
                            self.frame_cache.pop(0)
                        self.frame_cache.append(frame)
                        
                        # 清理检测结果
                        del results
                    
                    # 显示检测结果
                    cv2.imshow("Detection Results", display_frame)
                    key = cv2.waitKey(1)
                    if key == 27:  # ESC键退出
                        break
                    
                    del display_frame
                
                finally:
                    # 释放资源
                    del small_frame
                    if current_time - self.last_memory_check >= 5:  # 每5秒进行一次GC
                        gc.collect()

        except Exception as e:
            print(f"监控循环出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            gc.collect()

    def capture_window(self):
        """优化的窗口捕获函数"""
        if not self.target_window:
            return None
            
        hwnd = self.target_window[0]
        try:
            # 获取窗口大小
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            width = right - left
            height = bottom - top

            if width <= 0 or height <= 0 or width > 4096 or height > 2160:
                return None

            # 创建设备上下文
            try:
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, width, height)
                saveDC.SelectObject(saveBitMap)
                
                result = self.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
                
                if not result:
                    saveDC.BitBlt((0, 0), (width, height), mfcDC, (0, 0), win32con.SRCCOPY)

                bmpstr = saveBitMap.GetBitmapBits(True)
                img = np.frombuffer(bmpstr, dtype='uint8')
                img = img.reshape(height, width, 4)
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                # 清理资源
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                
                return img
                    
            except Exception as e:
                print(f"捕获窗口时发生错误: {e}")
                return None

        except Exception as e:
            print(f"捕获窗口失败: {e}")
            return None

    def get_class_color(self, class_id):
        """获取类别对应的颜色"""
        if class_id not in self.class_colors:
            self.class_colors[class_id] = tuple(np.random.randint(0, 255, 3).tolist())
        return self.class_colors[class_id]

    def draw_detections(self, frame, boxes):
        """在帧上绘制检测结果"""
        try:
            for box in boxes:
                # 确保坐标是整数
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # 获取颜色
                color = self.get_class_color(cls_id)
                
                # 绘制边框
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # 绘制标签
                label = f"{self.model.names[cls_id]} {conf:.2f}"
                (label_width, label_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                
                # 确保标签位置在图像范围内
                label_y = max(y1 - baseline - 5, label_height + baseline + 5)
                
                # 绘制标签背景
                cv2.rectangle(
                    frame,
                    (x1, label_y - label_height - baseline - 5),
                    (x1 + label_width, label_y),
                    color,
                    cv2.FILLED
                )
                
                # 绘制标签文本
                cv2.putText(
                    frame,
                    label,
                    (x1, label_y - baseline - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2
                )
            
            return frame
        except Exception as e:
            print(f"绘制检测结果时出错: {e}")
            return frame

    def play_alert(self):
        """播放警报声音"""
        try:
            if not self.is_playing:
                pygame.mixer.music.load(self.alert_sound)
                pygame.mixer.music.play()
                self.is_playing = True
        except Exception as e:
            print(f"播放警报失败: {e}")

    def stop_alert(self):
        """停止警报声音"""
        try:
            if self.is_playing:
                pygame.mixer.music.stop()
                self.is_playing = False
        except Exception as e:
            print(f"停止警报失败: {e}")

    def force_cleanup(self):
        """强制清理内存"""
        try:
            # 清空帧缓存
            self.frame_cache.clear()
            
            # 清理所有OpenCV窗口
            cv2.destroyAllWindows()
            
            # 强制执行多次垃圾回收
            for _ in range(3):
                gc.collect()
            
            # 压缩进程工作集
            ctypes.windll.psapi.EmptyWorkingSet(ctypes.windll.kernel32.GetCurrentProcess())
            
            # 重置检测间隔
            self.last_detection_time = time.time()
            
        except Exception as e:
            print(f"强制清理失败: {e}")

    def run(self):
        """主运行循环"""
        try:
            self.root.mainloop()
        finally:
            # 设置标志，防止在清理时访问已销毁的控件
            self.running = False
            
            # 停止监控线程
            if hasattr(self, 'monitor_thread') and self.monitoring:
                self.monitoring = False
                self.monitor_thread.join(timeout=1.0)
            
            # 清理资源
            self.cleanup_resources()
            
            # 清理OpenCV窗口和pygame
            try:
                cv2.destroyAllWindows()
            except:
                pass
                
            try:
                pygame.mixer.quit()
            except:
                pass

if __name__ == "__main__":
    monitor = DesktopMonitor()
    monitor.run()