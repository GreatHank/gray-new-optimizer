%% 参数设置
siz = 500;                  % 阵列大小
lambda = 480;               % 入射波长 (nm)
p_x = 850;                 % x方向周期 Λx
p_y = 850;                 % y方向周期 Λy

l = 145;                    % 方形边长 (nm)
w = 145;                    % 方形边宽 (nm)，这里与 l 相等也可以保留参数通用性

%% 载入梯度化矩阵 (从指定文件夹)
dataFolder = 'C:\Users\30358\PyCharmMiscProject\results_9channels_20251230_1233';  % 修改为你的文件夹路径

file_phdx = fullfile(dataFolder, 'phdx.csv');
file_phdy = fullfile(dataFolder, 'phdy.csv');

phdx = readmatrix(file_phdx);
phdy = readmatrix(file_phdy);

% 如果 CSV 文件是一维数据，则进行 reshape
if numel(phdx) == siz * siz
    phdx = reshape(phdx, siz, siz);
end
if numel(phdy) == siz * siz
    phdy = reshape(phdy, siz, siz);
end

% 将相位映射到 [0, 2π]
phdx = mod(phdx, 2 * pi);
phdy = mod(phdy, 2 * pi);

%% 1. 将 phdx/phdy 映射到纳米砖位移
displacement_x = mod(phdx / (2 * pi) * p_x, p_x);
displacement_y = mod(phdy / (2 * pi) * p_y, p_y);

%% 2. 输出 CIF 文件
outFolder = 'E:\zhangzhe\youhua\mask\ex1230';  % 修改为输出路径
if ~exist(outFolder, 'dir')
    mkdir(outFolder);
end
name = fullfile(outFolder, 'ex1230zz.cif');

fid = fopen(name, 'w');
fprintf(fid, 'DS 1 2 20;\r\n9 CELL0;\r\n');
fprintf(fid, 'L CMF;\r\n');

half_l = l / 2;
half_w = w / 2;

for i = 1:siz
    for j = 1:siz
        % 当前单元中心坐标
        x_center = (j - 0.5) * p_x + displacement_x(i, j);
        y_center = (siz + 0.5 - i) * p_y + displacement_y(i, j);

        % 方形四个顶点坐标 (顺时针)
        x1 = x_center - half_l;  y1 = y_center - half_w;
        x2 = x_center - half_l;  y2 = y_center + half_w;
        x3 = x_center + half_l;  y3 = y_center + half_w;
        x4 = x_center + half_l;  y4 = y_center - half_w;

        % 写入 CIF 格式
        fprintf(fid, 'P %.0f %.0f %.0f %.0f %.0f %.0f %.0f %.0f;\r\n', ...
            x1, y1, x2, y2, x3, y3, x4, y4);
    end
end

fprintf(fid, 'DF;\r\nC 1;\r\nE');
fclose(fid);

disp(['✅ CIF 文件生成完成，保存路径: ', name]);
