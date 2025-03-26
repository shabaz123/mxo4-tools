function result = read_mxofile(fname, chan)

% This MATLAB code reads .h5 files and returns an array of the selected channel data
% example usage:
%   fname = 'C:\MyFiles\Waveforms.h5';
%   chan = 1;
%   data = read_mxofile(fname, chan);
%   plot(data);

fid = H5F.open(fname);
file_info = hdf5info(fname, 'ReadAttributes', true);
groups = file_info.GroupHierarchy.Groups;
% search for the Waveform group
for i = 1:length(groups)
    if strcmp(groups(i).Name, '/Waveforms')
        group = groups(i);
        break;
    end
end

% search for the channel group, it will be called C1 or C2 etc..
channels = group.Groups;
% search for the channel, it will be named C<chan>, e.g. C1 if chan = 1
name = "/Waveforms/C"+chan;
for i = 1:length(channels)
    if strcmp(channels(i).Name, name)
        channel = channels(i);
        break;
    end
end

% search for the data group, it will be called C1 Data or C2 Data etc..
data = channel.Datasets;
name = name+"/C"+chan+" Data";
for i = 1:length(data)
    if strcmp(data(i).Name, name)
        dataset = data(i);
        break;
    end
end

% read the data
result = h5read(fname, dataset.Name);

end







