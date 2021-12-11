%% --KELOMPOK 6--------------------------
% Elyaser Ben Guno (1806195135)
% Thara Adiva Putri Candra (1806187594)
%% ---------------------------------------
%parameter yang dimasukkan oleh user
filename = evalin('base','filename');
numGauss = evalin('base','numGauss');
numTFrames = evalin('base','numTFrames');
minBGRatio = evalin('base','minBGRatio');
learnRate = evalin('base','learnRate');

% Membaca video
reader = VideoReader(filename);

% ForegroundDetector mengimplementasikan gaussian mixture model
% untuk mensegmentasi objek bergerak. outputnya adalah binary mask
detector = vision.ForegroundDetector('NumGaussians', numGauss, ...
	'NumTrainingFrames', numTFrames, 'MinimumBackgroundRatio', minBGRatio, ...
    'LearningRate', learnRate);

% Grup piksel foreground yang terkoneksi akan dideteksi sebagai blob
% oleh blob detection. lalu dihitung area, centroid dan bounding boxnya
blobAnalyser = vision.BlobAnalysis('BoundingBoxOutputPort', true, ...
	'AreaOutputPort', true, 'CentroidOutputPort', true, ...
	'MinimumBlobArea', 400);
% ---------------------------------------
tracks = struct(...
	'id', {}, ...
	'bbox', {}, ...
	'kalmanFilter', {}, ...
	'age', {}, ...
	'totalVisibleCount', {}, ...
	'consecutiveInvisibleCount', {});
% ---------------------------------------
nextId = 1;
numFrames = 0;

% Membuat objek untuk menyimpan video
v = VideoWriter('frame.avi');
w = VideoWriter('mask.avi');
afGMM = VideoWriter('afterGMM.avi');
open(v); open(w); open(afGMM);

%% ---------------------------------------
% Deteksi Objek
% ----------------------------------------
while hasFrame(reader)
    frame = readFrame(reader);    
    % Deteksi foreground.
    mask = detector.step(frame);
    afterGMM = mask;
    % Mengaplikasikan operasi morfologi untuk hilangkan noise
    % dan mengisi lubang-lubang
    mask = imopen(mask, strel('rectangle', [3,3]));  
    mask = imclose(mask, strel('rectangle', [15, 15]));
    mask = imfill(mask, 'holes');

    % Melakukan analisis blob untuk mencari komponen terhubung.
    [~, centroids, bboxes] = blobAnalyser.step(mask);
    
    %% ---------------------------------------
    % Prediksi lokasi baru dari track
    % ----------------------------------------
    for i = 1:length(tracks)
        bbox = tracks(i).bbox;

     	% Prediksi lokasi track saat ini menggunakan kalman filter
     	predictedCentroid = predict(tracks(i).kalmanFilter);

      	% Menggeser bounding box agar tengahnya berada di
    	% lokasi yang diprediksi
      	predictedCentroid = int32(predictedCentroid) - bbox(3:4) / 2;
     	tracks(i).bbox = [predictedCentroid, bbox(3:4)];
    end
    %% ---------------------------------------
    % Menetapkan deteksi pada track
    % ----------------------------------------
    nTracks = length(tracks);
  	nDetections = size(centroids, 1);

  	% Menghitung cost dari penetapan tiap deteksi pada track
  	cost = zeros(nTracks, nDetections);
  	for i = 1:nTracks
     	cost(i, :) = distance(tracks(i).kalmanFilter, centroids);
    end

  	% Menyelesaikan permasalahan assignment
	costOfNonAssignment = 20;
  	[assignments, unassignedTracks, unassignedDetections] = ...
     	assignDetectionsToTracks(cost, costOfNonAssignment);    
    
    %% ---------------------------------------
    % memperbarui assigned track
    % ----------------------------------------
    numAssignedTracks = size(assignments, 1);
  	for i = 1:numAssignedTracks
     	trackIdx = assignments(i, 1);
      	detectionIdx = assignments(i, 2);
      	centroid = centroids(detectionIdx, :);
     	bbox = bboxes(detectionIdx, :);

    	% Mengoreksi estimasi dari lokasi objek
      	% menggunakan deteksi baru
     	correct(tracks(trackIdx).kalmanFilter, centroid);

       	% Mengganti bounding box yang telah diprediksi
      	% dengan bounding box terdeteksi
      	tracks(trackIdx).bbox = bbox;

     	% Memperbarui track's age.
       	tracks(trackIdx).age = tracks(trackIdx).age + 1;

      	% Memperbarui visibilitas
       	tracks(trackIdx).totalVisibleCount = ...
      	tracks(trackIdx).totalVisibleCount + 1;
      	tracks(trackIdx).consecutiveInvisibleCount = 0;
    end
    
    %% ---------------------------------------
    % Memperbarui unassigned tracks
    % ----------------------------------------
    for i = 1:length(unassignedTracks)
     	ind = unassignedTracks(i);
       	tracks(ind).age = tracks(ind).age + 1;
     	tracks(ind).consecutiveInvisibleCount = ...
           	tracks(ind).consecutiveInvisibleCount + 1;
    end  
    
    %% ---------------------------------------
    % Menghapus track yang telah hilang
    % ----------------------------------------
  	invisibleForTooLong = 20;
 	ageThreshold = 8;

  	% Menghitung fraksi dari track's age
  	ages = [tracks(:).age];
   	totalVisibleCounts = [tracks(:).totalVisibleCount];
   	visibility = totalVisibleCounts ./ ages;

  	% Menemukan indeks dari track yang hilang
  	lostInds = (ages < ageThreshold & visibility < 0.6) | ...
      	[tracks(:).consecutiveInvisibleCount] >= invisibleForTooLong;

   	% Menghapus track yang hilang
  	tracks = tracks(~lostInds);
    
    %% ---------------------------------------
    % Membuat track baru
    % ----------------------------------------
    centroids = centroids(unassignedDetections, :);
  	bboxes = bboxes(unassignedDetections, :);

  	for i = 1:size(centroids, 1)
      	centroid = centroids(i,:);
       	bbox = bboxes(i, :);

      	% Membuat objek kalman filter
      	kalmanFilter = configureKalmanFilter('ConstantVelocity', ...
         	centroid, [200, 50], [100, 25], 100);

      	% Membuat track baru
     	newTrack = struct(...
          	'id', nextId, ...
           	'bbox', bbox, ...
          	'kalmanFilter', kalmanFilter, ...
          	'age', 1, ...
           	'totalVisibleCount', 1, ...
           	'consecutiveInvisibleCount', 0);

       	% Menambahkan track baru ke array track
       	tracks(end + 1) = newTrack;

       	% Inkremen next id
      	nextId = nextId + 1;
    end
    %% ---------------------------------------------------
    % Menambahkan hasil tracking & deteksi ke frame & mask
    % ----------------------------------------------------
  	% Convert frame dan mask ke uint8 RGB.
  	frame = im2uint8(frame);
   	mask = uint8(repmat(mask, [1, 1, 3])) .* 255;
    afterGMM = uint8(repmat(afterGMM, [1, 1, 3])) .* 255;

  	minVisibleCount = 8;
   	if ~isempty(tracks)

        % Noisy detections tend to result in short-lived tracks.
        % Only display tracks that have been visible for more than
        % a minimum number of frames.
       	reliableTrackInds = ...
        	[tracks(:).totalVisibleCount] > minVisibleCount;
       	reliableTracks = tracks(reliableTrackInds);

     	if ~isempty(reliableTracks)
          	% Mendapatkan bounding box
          	bboxes = cat(1, reliableTracks.bbox);

        	% Mendapatkan id
           	ids = int32([reliableTracks(:).id]);

            % Membuat label untuk objek 
          	labels = cellstr(int2str(ids'));
          	predictedTrackInds = ...
              	[reliableTracks(:).consecutiveInvisibleCount] > 0;
           	isPredicted = cell(size(labels));
         	isPredicted(predictedTrackInds) = {' predicted'};
           	labels = strcat(labels, isPredicted);

            % Menambahkan objek ke frame
          	frame = insertObjectAnnotation(frame, 'rectangle', ...
              	bboxes, labels);

         	% Menambahkan objek ke mask
          	mask = insertObjectAnnotation(mask, 'rectangle', ...
               	bboxes, labels);
        end
    end

    numFrames = numFrames + 1;
 	% Display the mask and the frame.
  	writeVideo(v, frame);
   	writeVideo(w, mask);
    writeVideo(afGMM, afterGMM);
end

assignin('base','numFrames',numFrames);
close(v); close(w); close(afGMM);