import nifty8.re as ift
from plot import plot, plot_nice

# Helper: init models
def init_model(m_pars, fl_pars, random_key=None):
    cf = ift.CorrelatedFieldMaker(m_pars["prefix"])
    cf.set_amplitude_total_offset(m_pars["offset_mean"], m_pars["offset_std"])
    cf.add_fluctuations(**fl_pars)
    field = cf.finalize()
    if random_key is not None:
        return ift.Vector(field.init(random_key))
    return None
    # print("model domain keys:", field.domain.keys())
    
# Helper: field and spectrum from parameter dictionaries + plotting
def eval_model(m_pars, fl_pars, dist, samples, title=None, name="", make_nice_plot=False):
    cf = ift.CorrelatedFieldMaker(m_pars["prefix"])
    cf.set_amplitude_total_offset(m_pars["offset_mean"], m_pars["offset_std"])
    cf.add_fluctuations(**fl_pars)
    field = cf.finalize()
    spectrum = cf.amplitude
    if not isinstance(samples, list):
        samples = [samples]
    field_realizations = [field(s) for s in samples]
    spectrum_realizations = [spectrum(s) for s in samples]
    plot(field_realizations, spectrum_realizations, dist, title)
    if make_nice_plot:
        plot_nice(field_realizations[0], spectrum_realizations[0], dist, path= "./", name=name)
    
