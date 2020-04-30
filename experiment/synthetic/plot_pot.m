function plot_pot(fn, fe)
    fn = squeeze(fn);
    fe = squeeze(fe);
    fe = reshape(fe, [3000,3000]);
    figure;plot(linspace(-30,30, 3000), fn(1,:))
    figure;plot(linspace(-30,30, 3000), fn(2,:))
    figure; mesh(linspace(-30,30,3000), linspace(-30,30,3000), fe)
end