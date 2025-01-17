# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
# *                                                         *
# * Author: Riccardo Finotello <riccardo.finotello@cea.fr>  *
# * Date:   2024-11-07                                      *
# *                                                         *
# * Environment variables for scripts and tools             *
# *                                                         *
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


export FRG_INSTALLDIR=$HOME/Code/frg-signal-detection

export FRG_CONFIGPATH=${FRG_INSTALLDIR}/configs
export FRG_DOCSPATH=${FRG_INSTALLDIR}/docs
export FRG_SRCPATH=${FRG_INSTALLDIR}/src
export FRG_SCRIPTSPATH=${FRG_INSTALLDIR}/scripts
export FRG_TESTPATH=${FRG_INSTALLDIR}/tests

export SCRATCHDIR=${SCRATCHDIR:-${HOME}/Datasets}
