import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
public class Tupian{
    //对彩色图像做频域高斯高通
    public static BufferedImage applyGaussianHighPassColor(BufferedImage src, double D0) {
        int w = src.getWidth();
        int h = src.getHeight();
        // 拆分 RGB 通道并归一化到 [0,1]
        double[][] r = new double[h][w];
        double[][] g = new double[h][w];
        double[][] b = new double[h][w];
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rgb = src.getRGB(x, y);
                int rr = (rgb >> 16) & 0xFF;
                int gg = (rgb >> 8) & 0xFF;
                int bb = (rgb) & 0xFF;
                r[y][x] = rr / 255.0;
                g[y][x] = gg / 255.0;
                b[y][x] = bb / 255.0;
            }
        }
        // 对三个通道分别做频域高通
        double[][] rOut = gaussianHighPassSingleChannel(r, D0);
        double[][] gOut = gaussianHighPassSingleChannel(g, D0);
        double[][] bOut = gaussianHighPassSingleChannel(b, D0);
        // 合成输出图像
        BufferedImage dst = new BufferedImage(w, h, BufferedImage.TYPE_INT_RGB);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int rr = clampToByte(rOut[y][x] * 255.0);
                int gg = clampToByte(gOut[y][x] * 255.0);
                int bb = clampToByte(bOut[y][x] * 255.0);
                int rgb = (rr << 16) | (gg << 8) | bb;
                dst.setRGB(x, y, rgb);
            }
        }
        return dst;
    }
    //对单通道做频域高斯高通,输入h x w 的 double 数组（0~1）
    private static double[][] gaussianHighPassSingleChannel(double[][] src, double D0) {
        int h = src.length;
        int w = src[0].length;
        // padding 到 2 的幂
        int H = nextPow2(h);
        int W = nextPow2(w);
        double[][] real = new double[H][W];
        double[][] imag = new double[H][W];
        // 预中心化，使低频搬到频谱中心
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double val = src[y][x];
                if (((x + y) & 1) == 1) {
                    val = -val;
                }
                real[y][x] = val;
                imag[y][x] = 0.0;
            }
        }
        fft2D(real, imag, false);
        // 构造高斯高通滤波器并相乘，频谱中心在 (H/2, W/2)
        double cy = H / 2.0;
        double cx = W / 2.0;
        double D0sq = D0 * D0;
        for (int y = 0; y < H; y++) {
            double dy = y - cy;
            for (int x = 0; x < W; x++) {
                double dx = x - cx;
                double D2 = dx * dx + dy * dy;
                double Hhp = 1.0 - Math.exp(-D2 / (2.0 * D0sq));
                double re = real[y][x];
                double im = imag[y][x];
                real[y][x] = re * Hhp;
                imag[y][x] = im * Hhp;
            }
        }
        fft2D(real, imag, true);
        double[][] out = new double[h][w];
        double scale = 1.0 / (H * W);
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                double val = real[y][x] * scale;
                if (((x + y) & 1) == 1) {
                    val = -val;
                }
                // 截断到 [0,1]
                if (val < 0.0) val = 0.0;
                if (val > 1.0) val = 1.0;
                out[y][x] = val;
            }
        }
        return out;
    }

    //先对每一行做 1D FFT，再对每一列做 1D FFT
    private static void fft2D(double[][] real, double[][] imag, boolean inverse) {
        int H = real.length;
        int W = real[0].length;
        // 行变换
        double[] rowRe = new double[W];
        double[] rowIm = new double[W];
        for (int y = 0; y < H; y++) {
            System.arraycopy(real[y], 0, rowRe, 0, W);
            System.arraycopy(imag[y], 0, rowIm, 0, W);
            fft1D(rowRe, rowIm, inverse);
            System.arraycopy(rowRe, 0, real[y], 0, W);
            System.arraycopy(rowIm, 0, imag[y], 0, W);
        }
        // 列变换
        double[] colRe = new double[H];
        double[] colIm = new double[H];
        for (int x = 0; x < W; x++) {
            for (int y = 0; y < H; y++) {
                colRe[y] = real[y][x];
                colIm[y] = imag[y][x];
            }
            fft1D(colRe, colIm, inverse);
            for (int y = 0; y < H; y++) {
                real[y][x] = colRe[y];
                imag[y][x] = colIm[y];
            }
        }
    }
    //一维 Cooley–Tukey FFT
    private static void fft1D(double[] real, double[] imag, boolean inverse) {
        int n = real.length;
        int logN = Integer.numberOfTrailingZeros(n);
        // 位反转排序
        for (int i = 0; i < n; i++) {
            int j = Integer.reverse(i) >>> (32 - logN);
            if (j > i) {
                double tmpRe = real[i];
                double tmpIm = imag[i];
                real[i] = real[j];
                imag[i] = imag[j];
                real[j] = tmpRe;
                imag[j] = tmpIm;
            }
        }
        // Danielson–Lanczos 部分
        for (int len = 2; len <= n; len <<= 1) {
            double angle = 2.0 * Math.PI / len * (inverse ? 1 : -1);
            double wlenRe = Math.cos(angle);
            double wlenIm = Math.sin(angle);
            for (int i = 0; i < n; i += len) {
                double wRe = 1.0;
                double wIm = 0.0;
                for (int j = 0; j < (len >> 1); j++) {
                    int u = i + j;
                    int v = i + j + (len >> 1);
                    double reU = real[u];
                    double imU = imag[u];
                    double reV = real[v];
                    double imV = imag[v];
                    // t = w * V
                    double tRe = wRe * reV - wIm * imV;
                    double tIm = wRe * imV + wIm * reV;
                    // U' = U + t
                    real[u] = reU + tRe;
                    imag[u] = imU + tIm;
                    // V' = U - t
                    real[v] = reU - tRe;
                    imag[v] = imU - tIm;
                    // w *= wlen
                    double nextWRe = wRe * wlenRe - wIm * wlenIm;
                    double nextWIm = wRe * wlenIm + wIm * wlenRe;
                    wRe = nextWRe;
                    wIm = nextWIm;
                }
            }
        }
    }
    private static int nextPow2(int n) {
        int p = 1;
        while (p < n) {
            p <<= 1;
        }
        return p;
    }
    private static int clampToByte(double v) {
        if (v < 0.0) v = 0.0;
        if (v > 255.0) v = 255.0;
        return (int) Math.round(v);
    }
    private static boolean isImageFile(File f) {
        String name = f.getName().toLowerCase();
        return name.endsWith(".png") || name.endsWith(".jpg") || name.endsWith(".jpeg") || name.endsWith(".gif");
    }
    public static void main(String[] args) throws IOException {
        double D0 = 200.0;   // 截止频率,越大则削弱的低频范围越大
        File input = new File("input");
        File output = new File("outputD"+(int)Math.round(D0));
        if (!input.exists() || !input.isDirectory()) {
            System.out.println("输入文件夹不存在： " + input.getAbsolutePath());
            return;
        }
        if (!output.exists()) {
            output.mkdirs();
        }
        File[] files = input.listFiles();
        for (File f : files) {
            if (!f.isFile() || !isImageFile(f)) {
                continue; // 跳过非图片
            }
            System.out.println("处理：" + f.getName());
            BufferedImage src = ImageIO.read(f);
            if (src == null) {
                System.out.println("  读取失败，跳过。");
                continue;
            }
            BufferedImage dst = applyGaussianHighPassColor(src, D0);
            String name = f.getName();
            int dot = name.lastIndexOf('.');
            String base = (dot > 0) ? name.substring(0, dot) : name;
            String ext = (dot > 0) ? name.substring(dot + 1) : "png";
            String outName = base + "_D" + Math.round(D0) + "." + ext;
            File outFile = new File(output, outName);
            ImageIO.write(dst, ext, outFile);
            System.out.println("  输出：" + outFile.getAbsolutePath());
        }
        System.out.println("全部处理完成。");
    }
}