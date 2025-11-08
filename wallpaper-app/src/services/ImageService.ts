// ImageService.ts
import { Directory, Filesystem } from '@capacitor/filesystem';
import { Capacitor, CapacitorHttp } from '@capacitor/core';
import { ApiConfig } from './APIService';

class ImageService {
  private blobToBase64(blob: Blob): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = reject;
      reader.onload = () => {
        const dataUrl = reader.result as string;
        const base64 = dataUrl.split(',')[1];
        resolve(base64);
      };
      reader.readAsDataURL(blob);
    });
  }

  async saveImage(data: unknown, apiConfig: ApiConfig): Promise<string[]> {
    const savedImageUris: string[] = [];

    if (apiConfig.response.image.content_type === 'URL') {
      const urls = (Array.isArray(data) ? data : [data]) as string[];
      for (const url of urls) {
        try {
          const response = await CapacitorHttp.get({ url, responseType: 'blob' });
          const responseData = response.data;

          let base64Data: string;
          if (Capacitor.isNativePlatform()) {
            base64Data = responseData;
          } else {
            base64Data = await this.blobToBase64(responseData as Blob);
          }

          const fileName = `wallpaper-${new Date().getTime()}.jpg`;
          const result = await Filesystem.writeFile({
            path: fileName,
            data: base64Data,
            directory: Directory.Data,
          });
          savedImageUris.push(result.uri);
        } catch (error) {
          console.error(`Failed to download or save image from URL: ${url}`, error);
        }
      }
    } else if (apiConfig.response.image.content_type === 'BINARY') {
      const items = Array.isArray(data) ? data : [data];
      for (const item of items) {
        try {
          let base64Data: string;
          if (Capacitor.isNativePlatform()) {
            base64Data = item as string;
          } else {
            base64Data = await this.blobToBase64(item as Blob);
          }

          const fileName = `wallpaper-${new Date().getTime()}.webp`;
          const result = await Filesystem.writeFile({
            path: fileName,
            data: base64Data,
            directory: Directory.Data,
          });
          savedImageUris.push(result.uri);
        } catch (error) {
          console.error('Failed to save binary image data', error);
        }
      }
    }

    return savedImageUris;
  }
}

export default new ImageService();
