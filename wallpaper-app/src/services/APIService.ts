// APIService.ts
import { CapacitorHttp, HttpOptions } from '@capacitor/core';

export interface ApiParameter {
  name?: string;
  friendly_name: string;
  type: 'integer' | 'boolean' | 'enum' | 'string' | 'list';
  value: string | number | boolean | string[];
  min_value?: number;
  max_value?: number;
  friendly_value?: string[];
  required?: boolean;
  enable?: boolean;
  split_str?: string;
}

export interface ApiResponseImage {
  path: string;
  is_list: boolean;
  content_type: 'URL' | 'BINARY';
  is_base64: boolean;
}

export interface ApiResponseOtherData {
    'one-to-one-mapping': boolean;
    path: string;
    friendly_name: string;
}

export interface ApiResponseOther {
    friendly_name: string;
    data: ApiResponseOtherData[];
}

export interface ApiResponse {
  image: ApiResponseImage;
  others?: ApiResponseOther[];
}

export interface ApiConfig {
  friendly_name: string;
  link: string;
  func: 'GET' | 'POST';
  APICORE_version: string;
  parameters: ApiParameter[];
  response: ApiResponse;
  intro?: string;
  icon?: string;
}

const API_FILES = [
  'acg_loliapi.api.json',
  'acg_uapi.api.json',
  'bing_today.api.json',
  'bq_uapi.api.json',
  'cihub_images.api.json',
  'furry_uapi.api.json',
  'jiangtokoto.api.json',
  'lolicon_api.api.json',
  'lorem_picsum.api.json',
  'paulzzh_touhou_project.api.json',
  'pollinations_ai.api.json',
  'qrcode_uapi.api.json',
];

class APIService {
  async loadApiConfigs(): Promise<ApiConfig[]> {
    const configs: ApiConfig[] = [];
    for (const file of API_FILES) {
      try {
        const response = await fetch(`/assets/EnterPoint/${file}`);
        if (!response.ok) {
          console.error(`Failed to load API config: ${file}`);
          continue;
        }
        const config = await response.json();
        configs.push(config);
      } catch (error) {
        console.error(`Error parsing API config: ${file}`, error);
      }
    }
    return configs;
  }

  async requestApi(apiConfig: ApiConfig, payload: Record<string, string | number | boolean | string[]>): Promise<unknown> {
    const constructApiUrl = (config: ApiConfig, params: Record<string, string | number | boolean | string[]>): string => {
      let url = config.link.replace(/\/$/, '');
      const queryParams: Record<string, string | number | boolean | string[]> = {};
      const pathParams: string[] = [];

      Object.keys(params).forEach(key => {
        const value = params[key];
        if (key.startsWith('_path_')) {
          pathParams.push(String(value));
        } else {
          queryParams[key] = value;
        }
      });

      if (pathParams.length > 0) {
        url += '/' + pathParams.join('/');
      }

      const urlObj = new URL(url);
      for (const key in queryParams) {
        const value = queryParams[key];
        if (value !== null && value !== undefined) {
            if (Array.isArray(value)) {
                const paramDef = config.parameters.find(p => p.name === key);
                const separator = paramDef?.split_str || '|';
                urlObj.searchParams.append(key, value.join(separator));
            } else {
                urlObj.searchParams.append(key, String(value));
            }
        }
      }
      return urlObj.toString();
    };

    const options: HttpOptions = {
      url: '', // Initialize
      method: apiConfig.func,
      headers: {
        'Content-Type': 'application/json',
        'Accept-Encoding': 'identity',
      },
    };

    if (apiConfig.response.image.content_type === 'BINARY') {
      options.responseType = 'blob';
    }

    if (apiConfig.func.toUpperCase() === 'GET') {
      options.url = constructApiUrl(apiConfig, payload);
    } else {
      options.url = apiConfig.link;
      options.data = payload;
    }

    try {
      const response = await CapacitorHttp.request(options);
      return response.data;
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  parseResponse(data: unknown, paths: string | string[]): unknown {
    if (typeof paths === 'string') {
      return this.resolvePath(data, paths.split('.').filter(p => p));
    } else {
      return paths.map(path => this.resolvePath(data, path.split('.').filter(p => p)));
    }
  }

  private resolvePath(obj: unknown, parts: string[]): unknown {
    if (parts.length === 0 || obj === null || obj === undefined) {
      return obj;
    }

    const currentPart = parts[0];
    const remainingParts = parts.slice(1);
    const indexMatch = currentPart.match(/\[(.*?)\]/);

    if (indexMatch) {
      const field = currentPart.substring(0, indexMatch.index);
      const indexExpr = indexMatch[1];

      let target: unknown = obj;
      if (field) {
        if (typeof target === 'object' && target !== null) {
            target = (target as Record<string, unknown>)[field];
        } else {
            return undefined;
        }
      }

      if (target === null || target === undefined) {
        return undefined;
      }

      if (indexExpr === '*') {
        if (!Array.isArray(target)) {
          return undefined;
        }
        return target.map(item => this.resolvePath(item, remainingParts));
      } else {
        const idx = parseInt(indexExpr, 10);
        if (Array.isArray(target) && !isNaN(idx) && idx < target.length) {
          return this.resolvePath(target[idx], remainingParts);
        }
        return undefined;
      }
    } else {
      if (typeof obj === 'object' && obj !== null && (obj as Record<string, unknown>)[currentPart] !== undefined) {
        return this.resolvePath((obj as Record<string, unknown>)[currentPart], remainingParts);
      }
      return undefined;
    }
  }
}

export default new APIService();
