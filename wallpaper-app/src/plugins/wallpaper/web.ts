import { WebPlugin } from '@capacitor/core';

import type { WallpaperPlugin, SetWallpaperOptions } from './definitions';

export class WallpaperWeb extends WebPlugin implements WallpaperPlugin {
  async set(options: SetWallpaperOptions): Promise<void> {
    console.log('Set wallpaper options', options);
    throw this.unimplemented('Setting wallpaper is not supported on the web.');
  }
}
