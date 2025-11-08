package io.ionic.starter.plugins.wallpaper;

import android.app.WallpaperManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.util.Log;

import com.getcapacitor.JSObject;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;

import java.io.InputStream;

@CapacitorPlugin(name = "Wallpaper")
public class WallpaperPlugin extends Plugin {

    @PluginMethod
    public void set(PluginCall call) {
        String path = call.getString("path");
        if (path == null) {
            call.reject("Image path is required");
            return;
        }

        try {
            Uri uri = Uri.parse(path);
            InputStream inputStream = getContext().getContentResolver().openInputStream(uri);
            Bitmap bitmap = BitmapFactory.decodeStream(inputStream);

            WallpaperManager wallpaperManager = WallpaperManager.getInstance(getContext());
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.N) {
                wallpaperManager.setBitmap(bitmap, null, true, WallpaperManager.FLAG_SYSTEM);
            } else {
                wallpaperManager.setBitmap(bitmap);
            }
            call.resolve();
        } catch (Exception e) {
            Log.e("WallpaperPlugin", "Error setting wallpaper", e);
            call.reject("Error setting wallpaper: " + e.getMessage());
        }
    }
}
