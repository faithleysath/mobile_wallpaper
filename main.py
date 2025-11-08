#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
壁纸生成器5 - 整合版本
所有模块已整合到单文件中，便于部署和运行

主要模块：
- Logger: 日志系统
- APIKernal: API请求和响应解析
- MainKernal: 辅助工具函数（图片下载、网络检测等）
- APICORE: API配置文件解析器
- GUI: 基于PySide6的图形界面
"""

# ==================== 标准库导入 ====================
import sys
import os
import asyncio
import gc
import base64
import traceback
import json
import re
import logging
import inspect
import mimetypes
import time
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from functools import wraps
from io import BytesIO
from urllib.parse import urlparse, unquote
from concurrent.futures import ThreadPoolExecutor

# ==================== 第三方库导入 ====================
import aiohttp
import aiofiles
import psutil
from PIL import Image
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QTextEdit, QSpinBox, QSlider,
    QCheckBox, QMessageBox, QScrollArea, QDialog, QSpacerItem, QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap, QImageReader
from qfluentwidgets import Flyout, InfoBarIcon, isDarkThemeMode

# ==================== Logger 模块 ====================

COLORS = {
    'DEBUG': '\033[92m',    # 绿色
    'INFO': '\033[34m',      # 蓝色
    'WARNING': '\033[33m',   # 黄色
    'ERROR': '\033[31m',     # 红色
    'CRITICAL': '\033[31;1m' # 红色加粗
}
RESET = '\033[0m'

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # 获取原始日志消息
        message = super().format(record)
        # 根据日志级别添加颜色
        levelname = record.levelname
        color = COLORS.get(levelname, '')
        
        colored_levelname = f"{color}{levelname}{RESET}"
        message = message.replace(levelname, colored_levelname)
        
        return message
    
class QAsyncFilter(logging.Filter):
    def filter(self, record):
        # 屏蔽qasync的DEBUG日志
        if record.name.startswith('qasync') and record.levelno <= logging.DEBUG:
            return False
        return True

def get_logger():
    # 获取调用栈信息
    frame = inspect.currentframe().f_back
    module = inspect.getmodule(frame)
    # 获取调用模块的名称
    module_name = module.__name__ if module else '__main__'
    logger_instance = logging.getLogger(module_name)
    
    # 为qasync设置更高的日志级别
    if module_name.startswith('qasync'):
        logger_instance.setLevel(logging.WARNING)
    
    return logger_instance

# 初始化根日志记录器
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)

# 移除所有现有处理器
for handler in root_logger.handlers[:]:
    root_logger.removeHandler(handler)

handler = logging.StreamHandler()
formatter = ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
handler.addFilter(QAsyncFilter())
root_logger.addHandler(handler)

# 创建全局logger实例
logger = get_logger()

# ==================== APIKernal 模块 ====================

def construct_api(api: str, payload: Optional[Dict[str, Any]] = None, split_str: Dict[str, str] = {}):
    """构造请求的URL和参数
    
    :param api: API端点URL
    :param payload: 请求负载(对于POST/PUT等)
    :param split_str: 对于是列表类型的请求负载，如果是 GET 方法，则将列表中的每个值用此字符连接（缺省为 '|'）"""
    
    url = api.rstrip('/').rstrip('?')
    none_params = []
    other_params = {}
    
    # 分离None键和其他参数
    for key, value in payload.items():
        if key is None:
            none_params.append(str(value))
        else:
            s = split_str.get(key, "")
            if not s:
                s = '|'
                
            if isinstance(value, list):
                other_params[key] = s.join(str(v) for v in value)
            else:
                other_params[key] = value
    
    # 构建最终URL
    if none_params:
        url += '/' + '/'.join(none_params)
    if other_params:
        url += '?' + '&'.join(f'{k}={v}' for k, v in other_params.items())
    
    logger.debug(f"构建的请求URL: {url}")
    return url

async def request_api(
    api: str,
    paths: Optional[Union[str, List[str]]] = None,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    payload: Optional[Dict[str, Any]] = None,
    split_str: Dict[str, str] = {},
    timeout: int = 15, 
    raw: bool = False,
    ssl_verify: bool = True  
) -> Any:
    """
    异步执行API请求并解析响应数据
    
    :param api: API端点URL
    :param paths: 要解析的一个或多个路径
    :param method: HTTP方法 (GET, POST, etc.)
    :param headers: 请求头
    :param payload: 请求负载(对于POST/PUT等)
    :param split_str: 对于是列表类型的请求负载，如果是 GET 方法，则将列表中的每个值用此字符连接（缺省为 '|'）
    :param timeout: 超时时间(秒)
    :param raw: True = 返回原始数据，False = 返回按照 paths 解析后的数据
    :param ssl_verify: 是否验证SSL证书，设置为False可禁用SSL验证
    :return: 解析后的数据
    """
    headers = headers or {}
    payload = payload or {}
    
    try:
        # 如果禁用SSL验证，创建一个不验证SSL的ClientSession
        connector = aiohttp.TCPConnector(ssl=ssl_verify) if not ssl_verify else None
        logger.warning(f"SSL 验证已被禁用，这可能会造成不安全的HTTPS连接！") if not ssl_verify else None
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=timeout),
            connector=connector
        ) as session:
            if method.upper() in ["GET", "HEAD"]:
                url = construct_api(api, payload, split_str)
                async with session.request(method, url, headers=headers) as response:
                    return await handle_response(response, paths, raw)
            else:
                async with session.request(method, api, headers=headers, json=payload) as response:
                    return await handle_response(response, paths, raw)
    except asyncio.TimeoutError:
        raise RuntimeError(f"API请求超时: 超过 {timeout} 秒")
    except aiohttp.ClientError as e:
        error_msg = f"API请求失败: {str(e)}"
        if hasattr(e, 'status') and e.status:
            error_msg += f" (状态码: {e.status})"
        raise RuntimeError(error_msg)

async def handle_response(response: aiohttp.ClientResponse, paths: Optional[Union[str, List[str]]] = None, raw = False) -> Tuple[Any, Any, Any]:
    """处理响应并返回解析后的数据（由 request_api 调用）"""

    try:
        content = await response.text()
    except UnicodeDecodeError:
        content = await response.text('latin1')
    
    logger.debug(f"API返回状态码: {response.status} {response.reason}")
    logger.debug(f"API返回内容: {content[:200]}...")
    if not response.ok:
        error_msg = f"API返回错误: {response.status} {response.reason}"
        if content:
            error_msg += f"\n错误详情: {content[:200]}..."
        raise RuntimeError(error_msg)
    
    if raw:
        return response, content, await response.read()
    
    # 解析JSON
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = content
    
    if not paths:
        return data, None, None
    
    # 解析指定的路径
    return parse_response(data, paths), None, None

def parse_response(data: Any, paths: Union[str, List[str]]) -> Any:
    """
    解析响应数据
    
    :param data: 要解析的数据 (通常是dict或list)
    :param paths: 单个路径字符串或路径字符串列表
    :return: 解析结果，结果类型根据路径和数据类型决定
    """
    # 核心：通过正则表达式，识别索引和切片语法
    index_pattern = re.compile(r'\[(.*?)\]')

    def resolve_path(obj: Any, parts: List[str]) -> Union[Any, List[Any]]:
        """递归解析路径：返回最自然的类型"""
        if not parts or obj is None:
            return obj
            
        current = parts[0]
        remaining = parts[1:]
        
        # 检查当前部分是否包含索引/切片语法
        match = index_pattern.search(current)
        if match:
            # 提取索引/切片表达式
            index_expr = match.group(1)
            # 获取字段名（索引前的部分）
            field = current[:match.start()].strip()
            
            # 如果字段名不为空，先访问该字段
            if field:
                if isinstance(obj, dict) and field in obj:
                    obj = obj[field]
                elif isinstance(obj, (list, tuple)) and field.isdigit():
                    obj = obj[int(field)]
            
                # 处理通配符 (*) 或索引/切片
                if index_expr == '*':
                    # 通配符处理，展开所有元素
                    if not isinstance(obj, (list, tuple)):
                        return None
                    
                    results = []
                    for item in obj:
                        result = resolve_path(item, remaining)
                        # 根据是否使用通配符决定结构
                        if '*' in current:  # 使用通配符时保持每个item的结构
                            results.append(result)
                        else:  # 非通配符情况扁平化
                            if isinstance(result, list):
                                results.extend(result)
                            else:
                                results.append(result)
                    return results
            elif ':' in index_expr:
                # 切片处理
                indices = index_expr.split(':')
                try:
                    start = int(indices[0]) if indices[0] else 0
                    end = int(indices[1]) if len(indices) > 1 and indices[1] else len(obj)
                    step = int(indices[2]) if len(indices) > 2 and indices[2] else 1
                    
                    results = []
                    for item in obj[start:end:step]:
                        result = resolve_path(item, remaining)
                        if isinstance(result, list):
                            results.extend(result)
                        else:
                            results.append(result)
                    return results
                except (ValueError, TypeError):
                    return None
            else:
                # 单个索引处理
                try:
                    idx = int(index_expr)
                    return resolve_path(obj[idx], remaining)
                except (ValueError, TypeError, IndexError):
                    return None
                
        # 处理常规路径
        if isinstance(obj, dict) and current in obj:
            return resolve_path(obj[current], remaining)
            
        # 解析为数组索引
        if isinstance(obj, (list, tuple)) and current.isdigit():
            try:
                idx = int(current)
                return resolve_path(obj[idx], remaining)
            except (ValueError, IndexError):
                return None
            
        # 分割点路径
        if '.' in current:
            sub_paths = current.split('.')
            return resolve_path(obj, sub_paths + remaining)
            
        return None
    
    # 处理单个路径或路径列表
    if isinstance(paths, str):
        # 对于单个路径，返回最自然的类型
        path_parts = [part.strip() for part in paths.split('.') if part.strip()]
        return resolve_path(data, path_parts)
    else:
        # 对于多个路径，返回结果列表
        return [resolve_path(data, [part.strip() for part in p.split('.') if part.strip()]) 
                for p in paths]

# ==================== MainKernal 模块 ====================

def internal_only(func):
    """装饰器：限制函数只能被本模块内的其他函数调用"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 获取调用栈信息
        caller_frame = inspect.currentframe().f_back
        caller_module = caller_frame.f_globals.get('__name__', '')
        
        # 仅允许本模块内的函数调用
        if caller_module != __name__:
            raise RuntimeError(f"{func.__name__} 是内部工具函数，禁止显式调用！")
        return func(*args, **kwargs)
    return wrapper

async def _requests_api(
    api: str, 
    path: Optional[Union[str, List[str]]] = None,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = {}, 
    payload: Optional[Dict[str, Any]] = {},
    timeout: int = 15
) -> Any:
    """
    简化的通用API请求函数
    
    :param api: API端点URL
    :param path: 要解析的路径（单个或多个）
    :param method: HTTP方法 (默认为GET)
    :param headers: 请求头
    :param payload: 请求负载
    :param timeout: 超时时间（秒）
    :return: 解析后的数据
    """
    a, _, _ = await request_api(
        api=api,
        paths=path,
        method=method,
        headers=headers,
        payload=payload,
        timeout=timeout
    )
    return a

async def TodayFortune() -> str:
    url = "https://v2.xxapi.cn/api/horoscope?type=aquarius&time=today"
    headers = {
        'User-Agent': 'xiaoxiaoapi/1.0.0 (https://xxapi.cn)'
    }

    try:
        data = await _requests_api(url, "data.todo", headers=headers)
        yi = data['yi']
        ji = data['ji']
        return f"{yi}\n{ji}"
    except:
        traceback.print_exc()
        return "无法获取今日运势"
    
async def Hitokota() -> str:
    content = ""
    try:
        content = str(await _requests_api(api="https://international.v1.hitokoto.cn/", path="hitokoto"))
    except:
        traceback.print_exc()
        print("\n\n")
        content = "请求失败。"
        
    return u"<html><head/><body><p><span style=\" font-size:12pt;\">{} </span></p><p align=\"right\"><span style=\" font-size:12pt;\">\u2014\u2014 \u4eca\u65e5\u56fe\u7247 </span></p></body></html>".format(content)
    
async def check_network() -> Union[float, bool]:
    try:
        logger.info("测试网络连接...")
        
        # 强制使用ASCII环境并设置ping目标
        env = os.environ.copy()
        env.update({
            'LC_ALL': 'C',
            'LANG': 'C',
            'LANGUAGE': 'C'
        })
        
        ping_target = 'cn.bing.com'
        timeout_sec = 5.0
        
        # Windows平台隐藏cmd窗口
        startupinfo = None
        if platform.system() == 'Windows':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = subprocess.SW_HIDE
            
        proc = await asyncio.create_subprocess_exec(
            'ping',
            '-n' if platform.system() == 'Windows' else '-c',
            '1',
            '-w' if platform.system() == 'Windows' else '-W',
            '1000',
            ping_target,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            startupinfo=startupinfo
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout_sec)
        except asyncio.TimeoutError:
            proc.kill()
            logger.warning(f"网络连接超时 (>{timeout_sec}秒)")
            return False
        
        # 检查命令执行状态
        if proc.returncode != 0:
            logger.warning(f"网络无法连接: {proc.returncode}")
            return False
            
        # 直接处理二进制输出，强制提取数字
        output_bytes = stdout
        output_ascii = ''.join(chr(b) if b < 128 else ' ' for b in output_bytes)
        logger.debug(f"Ping输出 (原始字节): {output_bytes!r}")
        logger.debug(f"Ping输出 (ASCII处理): {output_ascii}")
        
        # 简化延迟提取 - 只查找数字+ms模式
        match = re.search(r'(\d+)\s*ms', output_ascii)
        if match:
            try:
                latency = float(match.group(1))
                logger.debug(f"网络延迟: {latency}ms")
                return latency
            except ValueError:
                logger.debug(f"未知的网络延迟: {match.group(1)}")
                return 0.0
        
        return False
        
    except Exception:
        logger.exception("网络检测错误")
        return False    
    
def is_process_running(process_name):
    for proc in psutil.process_iter(['name']):
        if proc.info['name'] == process_name:
            return True
    return False

async def phrase_binary_images(response: aiohttp.ClientResponse, content, binary_data) -> List[bytes]:
    """解析二进制图片数据，支持multipart和单图"""
    images = []
    content_type = response.headers.get('Content-Type', '').lower()
    
    try:
        # 多图
        if 'multipart/' in content_type:
            images.extend(await _process_multipart(response, content))
        else:
            # 单图
            images.extend(await _process_single_image(response, binary_data))
            
        # 过滤非静态图片
        return [img for img in images if await _is_static_image(img)]
    except aiohttp.ClientConnectionError:
        logger.error("连接被服务器关闭，尝试重试...")
        raise
    except Exception as e:
        logger.error(f"解析图片数据失败: {str(e)}")
        logger.debug(traceback.format_exc())
        return []

@internal_only
async def _process_multipart(response, content):
    parts = []
    boundary_match = re.search(r'boundary="?([^";]+)"?', response.headers.get('Content-Type', ''))
    if not boundary_match:
        return []
    
    boundary = boundary_match.group(1)
    part_boundary = f'--{boundary}'
    end_boundary = f'{part_boundary}--'
    
    raw_parts = content.split(part_boundary)[1:]
    for raw_part in raw_parts:
        if raw_part.strip() == '--' or raw_part.startswith(end_boundary):
            continue
            
        header_body = raw_part.split('\r\n\r\n', 1)
        if len(header_body) < 2:
            continue
            
        header, body = header_body
        body = body.rstrip('\r\n')
        content_type = re.search(r'Content-Type:\s*([^\r\n]+)', header, re.IGNORECASE)
        if content_type:
            content_type = content_type.group(1).lower()
            if 'image/' in content_type:
                binary_body = body.encode('latin1')
                parts.append(binary_body)
    
    return parts

@internal_only
async def _process_single_image(response, binary_data):
    """处理单张图片的二进制数据"""
    content_type = response.headers.get('Content-Type', '').lower()
    if 'image/' in content_type or await _is_image_data(binary_data):
        return [binary_data]
    return []

@internal_only
async def _is_static_image(image_data):
    """动图排除"""
    try:
        img = Image.open(BytesIO(image_data))
        if img.format in ['GIF', 'WEBP', 'APNG']:
            try:
                img.seek(1)
                return False
            except EOFError:
                return True
        
        return img.format in ['JPEG', 'PNG', 'BMP', 'TIFF']
    
    except Exception:
        return False
    
@internal_only
async def _is_image_data(data):
    if len(data) < 12:
        return False
    
    # 魔数
    magic_numbers = {
        b'\xFF\xD8\xFF': 'JPEG',
        b'\x89PNG\r\n\x1a\n': 'PNG',
        b'GIF87a': 'GIF',
        b'GIF89a': 'GIF',
        b'BM': 'BMP',
        b'II*\x00': 'TIFF',
        b'MM\x00*': 'TIFF',
        b'\x00\x00\x01\x00': 'ICO',
    }
    
    for magic, fmt in magic_numbers.items():
        if data.startswith(magic):
            return True
    return False

async def download_images_binary(binary_data_list: List[bytes], save_path: str, max_workers: int = 4) -> Dict[str, bool]:
    """
    下载二进制图片(支持单张和多张)
    
    :param binary_data_list: 图片二进制数据列表
    :param save_path: 保存路径 (文件或目录)
    :param max_workers: 最大并发数
    :return: 字典 {文件名: 是否成功}
    """
    logger.info(f"保存二进制图片数据: {len(binary_data_list)}张 -> {save_path}")
    
    async def _save_single(data: bytes, path: str) -> bool:
        try:      
            async with aiofiles.open(path, 'wb') as f:
                await f.write(data)
            logger.info(f"图片保存成功: {path}")
            return True
        except Exception as e:
            logger.error(f"图片保存失败: {path} - {str(e)}")
            logger.debug(traceback.format_exc())
            return False
    
    if len(binary_data_list) == 1:
        if os.path.isdir(save_path):
            filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            save_path = os.path.join(save_path, filename)
        completeness = await _save_single(binary_data_list[0], save_path)
        return {save_path: completeness}
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    semaphore = asyncio.Semaphore(max_workers)
    results = {}
    tasks = []
    async def _save_with_semaphore(data, path):
        async with semaphore:
            return await _save_single(data, path)
        
    for i, data in enumerate(binary_data_list):
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}.jpg"
        path = os.path.join(save_path, filename)
        tasks.append(_save_with_semaphore(data, path))
    
    save_results = await asyncio.gather(*tasks, return_exceptions=True)
    for i, (data, result) in enumerate(zip(binary_data_list, save_results)):
        filename = f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i:03d}.jpg"
        path = os.path.join(save_path, filename)
        if isinstance(result, Exception):
            logger.error(f"图片保存异常: {path} - {str(result)}")
            logger.debug(traceback.format_exc())
            results[path] = False
        else:
            results[path] = result
    
    return results

async def download_images(urls: Union[str, List[str]], save_path: str, max_workers: int = 4, timeout: int = 30, retries: int = 3) -> Dict[str, Union[str, bool]]:
    """
    下载图片(支持单张和多张)
    
    :param urls: 图片URL或URL列表
    :param save_path: 保存路径 (文件或目录)
    :param max_workers: 最大并发数
    :param timeout: 超时时间(秒)
    :param retries: 重试次数
    :return: 字典 {URL: 图片路径（保存成功）/False（保存失败）}
    """
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    logger.info(f"下载图片: {urls} -> {save_path} , 超时: {timeout}秒, 重试: {retries}次")
    
    async def _download_single(session: aiohttp.ClientSession, url: str, path: str) -> bool:
        for attempt in range(retries):
            try:
                logger.info(f"开始下载图片: {url} -> {path} (尝试 {attempt + 1}/{retries})")
                async with session.get(url, headers=headers, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    async with aiofiles.open(path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                            downloaded += len(chunk)

                    logger.info(f"图片下载完成: {path}")
                    return True
                
            except Exception as e:
                logger.warning(f"图片下载失败 (尝试 {attempt + 1}/{retries}): {url} - {str(e)}")
                if attempt == retries - 1:
                    logger.error(f"图片下载最终失败: {url}")
                    logger.debug(traceback.format_exc())
                    return False
                await asyncio.sleep(1)
        return False
    
    def extract_filename(url, default_filename):
        origin_url = urlparse(url).path
        if not origin_url or origin_url.endswith('/'): 
            filename = default_filename
        else: 
            filename = unquote(os.path.basename(origin_url)).split('?')[0].split('#')[0]
        
        return (f"{filename}.jpg" if not "." in filename else filename).translate(str.maketrans(r'\/:*?"<>|', '_' * len(r'\/:*?"<>|')))
    
    if isinstance(urls, str):
        if os.path.isdir(save_path):
            filename = extract_filename(urls, f"image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
            save_path = os.path.join(save_path, filename)
        async with aiohttp.ClientSession() as session:
            completeness = await _download_single(session, urls, save_path)
            return {urls: save_path if completeness else False}
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    semaphore = asyncio.Semaphore(max_workers)
    results = {}
    
    async def _download_with_semaphore(session, url, path):
        async with semaphore:
            return await _download_single(session, url, path)
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, url in enumerate(urls):
            filename = extract_filename(url, f"image_{i}.jpg")
            path = os.path.join(save_path, filename)
            tasks.append(_download_with_semaphore(session, url, path))
        
        download_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, (url, result) in enumerate(zip(urls, download_results)):
            filename = extract_filename(url, f"image_{i}.jpg")
            path = os.path.join(save_path, filename)
            if isinstance(result, Exception):
                logger.error(f"图片下载异常: {path} - {str(result)}")
                logger.debug(traceback.format_exc())
                if os.path.exists(path):
                    os.remove(path)
                results[url] = False
            else:
                results[url] = path if result else False
    
    return results

def adaptive_link_splitter(text) -> List[str]:
    """链接解析"""
    common_separators = ["\n", "\n\n", " ", "  ", "    ", ",  ", ", ", ",", ";", "|"]
    
    for sep in common_separators:
        if sep in text:
            links = text.split(sep)
            valid_links = [link.strip() for link in links if link.strip()]
            
            if len(valid_links) > 0 and sum(is_link_like(link) for link in valid_links) / len(valid_links) > 0.75:
                return valid_links
    
    return re.split(r"\s+", text)

def is_link_like(link):
    clean_link = link.strip()
    conditions = [
        clean_link.startswith(("http://", "https://", "ftp://", "www.")), 
        "." in clean_link,
        " " not in clean_link, 
        len(clean_link) >= 8 
    ]
    return sum(conditions) >= 3

def adaptive_base64_extractor(text: str) -> Tuple[List[bytes], List[str]]:
    """
    base64解析
    :param text: 所有base64字符串
    :return: 解码后的二进制数据列表, 原始base64字符串列表
    """
    
    common_separators = ["\n", "\n\n", " ", "  ", "    ", ",  ", ", ", ",", ";", "|", "||", "::", ":", "\\", "/", "//"]
    for sep in common_separators:
        if sep in text:
            parts = text.split(sep)
            valid_parts = [part.strip() for part in parts if part.strip()]
            
            # 检查 Base64 比例
            base64_count = sum(is_base64_like(part) for part in valid_parts)
            if base64_count > 0 and base64_count / len(valid_parts) > 0.75:
                decoded_list = []
                encoded_list = []
                for part in valid_parts:
                    try:
                        decoded = base64.b64decode(part)
                        decoded_list.append(decoded)
                        encoded_list.append(part)
                    except:
                        pass
                return decoded_list, encoded_list
    
    base64_pattern = r'(?:[A-Za-z0-9+/]{4})*(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?'
    matches = re.findall(base64_pattern, text)
    valid_matches = [match for match in matches if is_base64_like(match)]
    
    decoded_list = []
    encoded_list = []
    for match in valid_matches:
        try:
            decoded = base64.b64decode(match)
            decoded_list.append(decoded)
            encoded_list.append(match)
        except:
            pass
    
    return decoded_list, encoded_list

def is_base64_like(s: str) -> bool:
    if len(s) < 4 or len(s) % 4 != 0:
        return False
    
    # 字符集检查
    valid_chars = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=")
    if not all(char in valid_chars for char in s):
        return False
    
    # 填充检查 ( = < 2 )
    if s.endswith('='):
        if s.count('=') > 2 or s[-1] != '=' or (len(s) > 1 and s[-2] == '=' and s[-1] != '='):
            return False
    
    try:
        base64.b64decode(s)
        return True
    except:
        return False

def get_internal_dir() -> str:
    if getattr(sys, 'frozen', False):
        base = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        if base.rstrip(os.path.sep).endswith("_internal"):
            return base
        
        if hasattr(sys, '_MEIPASS'):
            return os.path.join(base, '_internal')
        return os.path.join(base, '_internal') if os.path.exists(os.path.join(base, '_internal')) else base
    else:
        return os.path.dirname(os.path.realpath(sys.argv[0]))
    
def get_config_dir() -> str:
    config_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    if getattr(sys, 'frozen', False) or False:
        if sys.platform.startswith('win'):
            config_dir = str(Path.home() / f'AppData{os.sep}Roaming{os.sep}wallpaper-generator-next')
        elif sys.platform.startswith('linux'):
            config_dir = str(Path.home() / f'.config{os.sep}wallpaper-generator-next')
        elif sys.platform.startswith('darwin'):
            config_dir = str(Path.home() / f'Library{os.sep}Application Support{os.sep}wallpaper-generator-next')
        else:
            config_dir = str(Path.home() / f'.config{os.sep}wallpaper-generator-next')
    
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
        
    return config_dir

# ==================== APICORE 模块 ====================

def parse_image(response_data: Dict[str, Any]) -> Dict[str, Any]:
    """Parse image configuration from response data."""
    return response_data.get('image', {})

class Response:
    """响应包装器(支持插件化解析)"""
    
    def __init__(self, response_data: Dict[str, Any]):
        self.data = response_data
        # 内联 image parser
        self.image = lambda: parse_image(self.data)
                
    def others(self) -> List[Dict[str, Any]]:
        """获取配置中的其他响应配置"""
        return self.data.get('others', [])

class APICORE:
    """APICORE 规范的 API 配置文件解析器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        
    def init(self) -> 'APICORE':
        """初始化并检查配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
                
            # 验证这个配置文件是否达到基础的合格标准
            if not all(key in self.config for key in ['friendly_name', 'link', 'func', 'APICORE_version', 'parameters','response']):
                raise ValueError("配置文件中缺少部分必填字段 ('friendly_name', 'link', 'func', 'APICORE_version', 'parameters','response')")
                
            if self.config['APICORE_version'] != '1.0':
                raise ValueError("不受支持的 APICORE 版本")
                
            return self
            
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {str(e)}")
        except Exception as e:
            raise ValueError(f"配置文件验证失败: {str(e)}")
    
    def friendly_name(self) -> str:
        """获取 API 友好名称"""
        return self.config['friendly_name']
    
    def intro(self) -> str:
        """获取 API 描述"""
        return self.config.get('intro', '')
    
    def icon(self) -> str:
        """获取 API 的图标图片链接"""
        return self.config.get('icon', '')
    
    def link(self) -> str:
        """获取 API 链接"""
        return self.config['link']
    
    def func(self) -> str:
        """获取 API 调用方法"""
        return self.config['func']
    
    def version(self) -> str:
        """获取 APICORE 版本"""
        return self.config['APICORE_version']
    
    def parameters(self) -> List[Dict[str, Any]]:
        """获取参数配置列表"""
        return self.config.get('parameters', [])
    
    def response(self) -> Response:
        """获取响应配置列表"""
        return Response(self.config.get('response', {}))

# ==================== 全局变量 ====================

_generated = []  # 生成的图片路径列表
_other_responses = []  # 其他响应列表
images_generated = []  # 生成的图片对象或图片路径列表

# ==================== GUI 模块 ====================

class SimpleDialog(QDialog):
    """简单的对话框，用于显示单个图片"""
    def __init__(self, image, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generated Image")
        self.setMinimumSize(800, 600)
        
        layout = QVBoxLayout(self)
        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)
        
        # 根据图片大小调整显示
        if isinstance(image, str):  # 图片路径
            pixmap = QPixmap(image)
        else:  # 图片对象
            pixmap = QPixmap.fromImage(image)
            
        # 缩放图片以适应窗口，保持宽高比
        scaled_pixmap = pixmap.scaled(
            800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        
        self.label.setPixmap(scaled_pixmap)
        layout.addWidget(self.label)

def Construct_control():
    """从 EnterPoint 文件夹加载所有 .api.json 配置文件"""
    path = os.path.join(get_config_dir(), "EnterPoint")
    api_json_files = []
    if not os.path.isdir(path):
        os.mkdir(path)
        logger.info(f"Created {path}")
        
    logger.info(f"Checking {path} for API JSON files")
    for file in os.listdir(path):
        full_path = os.path.join(path, file)
        logger.debug(f"Checking {full_path}")
        if os.path.isfile(full_path) and full_path.endswith('.api.json') and not file.startswith('_'):
            try:
                api_json_files.append(APICORE(full_path).init())
            except Exception as e:
                logger.error(f"Failed to initialize API config {file}: {str(e)}")
    
    return api_json_files

async def on_api_start(cfg: APICORE, payload: dict):
    """从配置文件和构造好的参数请求 API"""
    try:
        split_str = {}
        download_path = os.path.abspath(os.path.join(get_config_dir(), "temp_dir"))
        if not os.path.isdir(download_path):
            os.makedirs(download_path)
            logger.info(f"Created download directory: {download_path}")
        
        for param in cfg.parameters():
            for key, value in param.items():
                if key == "split_str" and value:
                    split_str[key] = str(value)
                    break
        
        gc.collect()
        logger.debug(f"请求参数为列表的分隔符映射: {split_str}")
        binary_phrase = (str(cfg.response().image().get('content_type', "URL")).upper() == "BINARY")
        
        # 调用API
        r, t, c = await request_api(
            cfg.link(),
            "",
            cfg.func(), 
            payload=payload,
            timeout=30,
            split_str=split_str, 
            raw=binary_phrase, 
            ssl_verify=False,
        )
        
        # 逻辑最复杂的部分，解析响应
        result, response = None, []
        if binary_phrase:  # 二进制
            if not cfg.response().image().get('is_base64', False):  # 不是base64编码的单/多图片
                response = await phrase_binary_images(r, t, c)
            else:  # base64编码的单/多图片
                path = str(cfg.response().image().get('path', ''))
                if not path:
                    raise ValueError("API 配置文件中返回结果中缺少图片路径")
                    
                if isinstance(r, str):  # 纯文本的 base64 图片
                    response, _ = adaptive_base64_extractor(str(r))
                elif isinstance(r, list):
                    response, _ = adaptive_base64_extractor("\n".join(r))
                elif not cfg.response().image().get('is_list', True):  # 不在列表里的 base64 图片
                    response, _ = adaptive_base64_extractor(parse_response(r, path))
                else:
                    response = parse_response(r, path)
                    response[:] = [base64.b64decode(str(item)) for item in response]
                    
            result = await download_images_binary(
                response, 
                download_path,
            )
                
        else:  # URL
            logger.info(f"结果返回: {str(r)} {type(r)}")
            path = str(cfg.response().image().get('path', ''))
            if not path:
                raise ValueError("API 配置文件中返回结果中缺少图片路径")
            
            if isinstance(r, str):  # 纯文本的 URL
                response = adaptive_link_splitter(str(r))
                logger.debug(f"文本链接提取格式化: {response}")
            elif isinstance(r, list):
                response = adaptive_link_splitter("\n".join(r))
            elif not cfg.response().image().get('is_list', True):  # 不在列表里的URL
                response = adaptive_link_splitter(parse_response(r, path))
            else:
                response = parse_response(r, path)
                
            if cfg.response().image().get('is_base64', False):  # 用base64编码的URL
                response[:] = [base64.b64decode(str(item)).decode('utf-8') for item in response]
                    
            result = await download_images(
                response,
                download_path,
                retries=1,
                timeout=30
            )

        # 其他响应
        if len(cfg.response().others()) > 0:
            for other in cfg.response().others():
                name = other.get('friendly_name', '')
                data = other.get('data', [])
                if not name or not data or len(data) == 0:
                    continue

                field = {name: {}}
                for d in data:
                    if not bool(d.get('one-to-one-mapping', True)):
                        field[name].update({f"{d['friendly_name']}-no-one-to-one-mapping": parse_response(r, d['path'])})
                    else:
                        content = parse_response(r, d['path'])
                        if isinstance(content, list) and len(content) == len(response):
                            # result中所有False值都代表图片下载失败，需要从content中删除
                            false_indices = []
                            for idx, url in enumerate(response):
                                if url in result and result[url] is False:
                                    false_indices.append(idx)
                            
                            for idx in sorted(false_indices, reverse=True):
                                if idx < len(content):
                                    del content[idx]
                                    
                        field[name].update({d['friendly_name']: content})

                _other_responses.append(field)

        logger.info(f"其他响应: {_other_responses}")
        
        # 在主线程中完成操作
        QTimer.singleShot(0, lambda: on_api_complete(
            True,
            f"{cfg.friendly_name()} 生成成功",
            result if result else {},
            response, 
            cfg
        ))
        
    except Exception as e:
        logger.error(f"API请求失败: {str(e)}")
        logger.debug(traceback.format_exc())
        
        try:
            error_str = str(e)
        except Exception:
            error_str = repr(e)
        QTimer.singleShot(0, lambda: on_api_complete(
            False, 
            error_str,  
            {}, 
            [], 
            cfg
        ))

def on_api_complete(success, message, response: dict, links: list, cfg: APICORE):
    global _generated, images_generated
    logger.info(f"API请求完成，成功: {success}, 信息: {message}, 结果: {response}")
    
    # 清空之前的结果
    _generated = []
    images_generated = []
    
    for k, v in response.items():
        if v is not False:
            if v == True:
                _generated.append(k)
            else:
                _generated.append(v)
                
    gc.collect()
    if success and len(_generated) > 0:
        QImageReader.setAllocationLimit(512)
        logger.info(message)
        for path in _generated:
            reader = QImageReader(path)
            if reader.size().width() > 4096 or reader.size().height() > 4096:
                reader.setScaledSize(reader.size().scaled(2048, 2048, Qt.AspectRatioMode.KeepAspectRatio))
            
            image = reader.read()
            if image.isNull():
                images_generated.append(path)
            else:
                images_generated.append(image)
                
        # 显示生成的图片，每个图片一个dialog
        for image in images_generated:
            dialog = SimpleDialog(image)
            dialog.exec()
    else:
        logger.error("没有生成任何图片")
        QMessageBox.critical(None, "错误", message)

class ParamWidget(QWidget):
    """动态参数控件基类"""
    def __init__(self, param, parent=None):
        super().__init__(parent)
        self.param = param
        self.layout = QVBoxLayout(self)
        self.param_name = param.get('name') or f"param_{id(param)}"

class IntegerParamWidget(ParamWidget):
    """整数类型参数控件"""
    def __init__(self, param, parent=None):
        super().__init__(param, parent)
        
        # 创建标题
        title_layout = QHBoxLayout()
        self.title = QLabel(param.get('friendly_name', ''))
        title_layout.addWidget(self.title)
        
        # 创建滑块和数值选择器
        control_layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(param.get('min_value', 1))
        self.slider.setMaximum(param.get('max_value', 100))
        self.slider.setValue(param.get('value', 1))
        
        self.spinbox = QSpinBox()
        self.spinbox.setMinimum(param.get('min_value', 1))
        self.spinbox.setMaximum(param.get('max_value', 100))
        self.spinbox.setValue(param.get('value', 1))
        
        # 连接信号
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)
        
        control_layout.addWidget(self.slider)
        control_layout.addWidget(self.spinbox)
        
        self.layout.addLayout(title_layout)
        self.layout.addLayout(control_layout)
    
    def get_value(self):
        return self.spinbox.value()

class BooleanParamWidget(ParamWidget):
    """布尔类型参数控件"""
    def __init__(self, param, parent=None):
        super().__init__(param, parent)
        
        self.checkbox = QCheckBox(param.get('friendly_name', ''))
        self.checkbox.setChecked(bool(param.get('value', False)))
        self.layout.addWidget(self.checkbox)
    
    def get_value(self):
        return str(self.checkbox.isChecked()).lower()

class EnumParamWidget(ParamWidget):
    """枚举类型参数控件"""
    def __init__(self, param, parent=None):
        super().__init__(param, parent)
        
        layout = QHBoxLayout()
        self.title = QLabel(param.get('friendly_name', ''))
        
        self.combobox = QComboBox()
        opts = param.get('friendly_value')
        if opts and len(opts) == len(param.get('value', [])):
            self.combobox.addItems(opts)
            self.friendly_values = opts
        else:
            self.combobox.addItems(param.get('value', []))
            self.friendly_values = None
        self.combobox.setCurrentIndex(0)
        
        layout.addWidget(self.title)
        layout.addWidget(self.combobox)
        self.layout.addLayout(layout)
        
        self.raw_values = param.get('value', [])
    
    def get_value(self):
        index = self.combobox.currentIndex()
        if index < len(self.raw_values):
            return self.raw_values[index]
        return None

class StringParamWidget(ParamWidget):
    """字符串类型参数控件"""
    def __init__(self, param, parent=None):
        super().__init__(param, parent)
        
        self.title = QLabel(param.get('friendly_name', ''))
        self.textedit = QTextEdit()
        self.textedit.setText(param.get('value', ''))
        
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.textedit)
    
    def get_value(self):
        return self.textedit.toPlainText()

class ListParamWidget(ParamWidget):
    """列表类型参数控件"""
    def __init__(self, param, parent=None):
        super().__init__(param, parent)
        
        split_str = param.get('split_str', '|')
        title_text = f"{param.get('friendly_name', '')} （用 {split_str} 分隔）"
        self.title = QLabel(title_text)
        
        value = param.get('value', [])
        if isinstance(value, list):
            text = split_str.join(value)
        else:
            text = str(value)
        
        self.textedit = QTextEdit()
        self.textedit.setText(text)
        
        self.layout.addWidget(self.title)
        self.layout.addWidget(self.textedit)
        
        self.split_str = split_str
    
    def get_value(self):
        text = self.textedit.toPlainText()
        return text.split(self.split_str) if text else []

class MainWindow(QMainWindow):
    """主窗口类"""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("壁纸生成器5 (Demo: 2025.10.21) 内部使用禁止外传")
        self.setGeometry(100, 100, 900, 700)
        
        # 创建中央部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # 创建主布局
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # 创建API选择下拉菜单
        self.api_layout = QHBoxLayout()
        self.api_label = QLabel("选择API：")
        self.api_combobox = QComboBox()
        self.api_layout.addWidget(self.api_label)
        self.api_layout.addWidget(self.api_combobox)
        self.main_layout.addLayout(self.api_layout)
        
        # 创建参数区域滚动视图
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.param_container = QWidget()
        self.param_layout = QVBoxLayout(self.param_container)
        self.scroll_area.setWidget(self.param_container)
        self.main_layout.addWidget(self.scroll_area)
        
        # 创建请求按钮
        self.request_button = QPushButton("请求API")
        self.request_button.clicked.connect(self.on_request_clicked)
        self.main_layout.addWidget(self.request_button)
        
        # 加载API配置
        self.api_configs = Construct_control()
        self.current_config = None
        self.param_widgets = []
        
        # 填充API下拉菜单
        for config in self.api_configs:
            self.api_combobox.addItem(config.friendly_name())
        
        # 连接下拉菜单信号
        self.api_combobox.currentIndexChanged.connect(self.on_api_changed)
        
        # 如果有配置，默认选择第一个
        if self.api_configs:
            self.on_api_changed(0)
    
    def on_api_changed(self, index):
        """当选择的API改变时，重新构建参数UI"""
        if index < 0 or index >= len(self.api_configs):
            return
        
        # 清空之前的参数控件
        for widget in self.param_widgets:
            widget.setParent(None)
        self.param_widgets.clear()
        
        # 获取当前配置
        self.current_config = self.api_configs[index]
        logger.info(f"当前选择的API配置: {self.current_config.friendly_name()}")
        
        # 根据参数类型创建对应的控件
        control_classes = {
            "integer": IntegerParamWidget,
            "boolean": BooleanParamWidget,
            "enum": EnumParamWidget,
            "string": StringParamWidget,
            "list": ListParamWidget,
        }
        
        for param in self.current_config.parameters():
            param_type = str(param.get('type', '')).lower()
            if param_type in control_classes and bool(param.get('enable', True)):
                widget = control_classes[param_type](param)
                self.param_layout.addWidget(widget)
                self.param_widgets.append(widget)
        
        # 添加垂直间隔，确保底部有空间
        self.param_layout.addItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))
    
    def build_payload(self):
        """构建API请求的payload参数"""
        if not self.current_config:
            logger.error("当前没有选择的API配置")
            return {}
        
        payload = {}
        # 首先处理启用的参数（从UI获取值）
        for i, widget in enumerate(self.param_widgets):
            param = widget.param
            param_name = param.get('name')
            logger.debug(f"处理参数[{i}]: name={param_name}")
            
            # 获取值
            value = widget.get_value()
            
            # 确保所有非None的值都被添加，或者参数是必填的
            if value is not None or bool(param.get('required', False)):
                payload[param_name] = value
        
        # 处理未启用但需要传递默认值的参数
        for i, param in enumerate(self.current_config.parameters()):
            if not bool(param.get('enable', True)):
                param_name = param.get('name')
                # 关键修复：对于没有name的未启用参数，也使用param_{索引}作为替代
                if not param_name:
                    param_name = f"param_{i}"
                    logger.debug(f"未启用参数没有name，使用替代名称: {param_name}")
                
                if param_name and param_name not in payload:
                    payload[param_name] = param.get('value')
                    logger.debug(f"未启用参数 {param_name} 的默认值: {param.get('value')}")
        
        logger.debug(f"构建的完整请求参数: {payload}")
        return payload
    
    def on_request_clicked(self):
        """请求按钮点击事件"""
        if not self.current_config:
            QMessageBox.warning(self, "警告", "请先选择一个API配置")
            return
        
        try:
            # 禁用按钮防止重复点击
            self.request_button.setEnabled(False)
            self.request_button.setText("请求中...")
            
            # 构建请求参数
            payload = self.build_payload()
            logger.debug(f"构建的请求参数: {payload}")
            logger.info(f"开始: {self.current_config.friendly_name()} 模式生成...")
            
            # 创建并运行异步任务
            try:
                # 检查是否已有运行中的事件循环
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # 如果有运行中的循环，则创建任务
                    loop.create_task(on_api_start(self.current_config, payload))
                else:
                    # 否则运行任务
                    loop.run_until_complete(on_api_start(self.current_config, payload))
            except RuntimeError:
                # 如果没有事件循环，则创建一个新的
                asyncio.run(on_api_start(self.current_config, payload))
            
        except Exception as e:
            logger.error(f"生成过程中发生错误: {str(e)}")
            logger.debug(traceback.format_exc())
            QMessageBox.critical(self, "错误", f"生成过程中发生错误: {str(e)}")
        finally:
            # 恢复按钮状态
            self.request_button.setEnabled(True)
            self.request_button.setText("请求API")

# ==================== 主程序入口 ====================

def main():
    """主程序入口"""
    # 确保中文显示正常
    QApplication.setApplicationName("壁纸生成器")
    
    # 创建应用程序实例
    app = QApplication(sys.argv)
    
    # 设置中文字体
    font = app.font()
    font.setFamily("SimHei")  # 使用黑体作为默认字体
    app.setFont(font)
    
    # 创建并显示主窗口
    window = MainWindow()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
