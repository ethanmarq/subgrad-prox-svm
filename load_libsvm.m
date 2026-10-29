function [y, X] = load_libsvm(filename, n_features)
      fid = fopen(filename, 'r');
      raw = textscan(fid, '%s', 'Delimiter', '\n');
      fclose(fid);
      lines = raw{1};

      n = numel(lines);
      y = zeros(n, 1);
      X = zeros(n, n_features);

      for i = 1:n
          parts = strsplit(lines{i});
          y(i) = str2double(parts{1});
          for j = 2:numel(parts)
              if isempty(parts{j}), continue; end
              kv = strsplit(parts{j}, ':');
              col = str2double(kv{1});
              val = str2double(kv{2});
              X(i, col) = val;
          end
      end
  end
