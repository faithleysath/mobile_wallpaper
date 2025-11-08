export interface WallpaperPlugin {
  /**
   * Set the device wallpaper.
   *
   * @since 1.0.0
   */
  set(options: SetWallpaperOptions): Promise<void>;
}

export interface SetWallpaperOptions {
  /**
   * The path of the image file to set as wallpaper.
   * The path should be a native file path.
   */
  path: string;
}
