# -*- coding: utf-8 -*-
"""
@author: Joey Davis <jhdavis@mit.edu> jhdavislab.org
@version: 0.0.4
"""

import subprocess
import argparse
import time
import pandas as pd
import os
import shutil
import glob
import math
from pysodist.utils import utilities
import tarfile
import sys

log = utilities.log


def wait(processes, limit, wait_time, logfile=None):
    while len(processes) >= limit:
        log('waiting ' + str(wait_time) + ' seconds to check on processes...', logfile)
        time.sleep(wait_time)
        processes.difference_update([
            p for p in processes if p.poll() is not None])
        log(str(len(processes)) + ' are still running...', logfile)
    return processes


def run_fortran_isodist(all_in_files, base_command, threads=1, wait_time=120,
                        log_suffix='.log', err_suffix='.err', logfile=None, pythonic=False, corr_atom = None):
    """ Wrapper function to actually execute the fortran code. Should run multiple instances in parallel,
    and should work in Windows and linux.
    :param all_in_files: list of all of the '.in' files to be processed.
    :param base_command: string with the command to execute - should be full path to isodist executable.
    :param threads:  number of instances of isodist to spawn in parallel (default = 1).
    Typically 1 fewer than the  number of cores on the processor
    :param wait_time: how often to poll whether an isodist instance has finished. Default = 120 seconds.
    :param log_suffix: suffix for the log files saved by isodist.
    :param err_suffix: suffix for the err files saved by isodist.
    :param logfile: path to file to log
    :param pythonic: use the pythonic version of isodist
    :param corr_atom: atom in model file for which the correlation groups correspond to

    :return: a list of the csv files created as a result of fitting
    """
    processes = set()
    for name in all_in_files:
        working_dir = '/'.join(name.split('/')[:-1]) + '/'
        in_file = name.split('/')[-1]
        log('running command: ' + base_command + ' ' + in_file + ' in directory: ' + working_dir, logfile)
        with open(name.split('.in')[0] + log_suffix, 'w') as log_file, open(name.split('.in')[0] + err_suffix,
                                                                            'w') as err_file:
            if pythonic:
                if corr_atom is not None:
                    correlate_command = "--correlate"
                    processes.add(subprocess.Popen([sys.executable, base_command, in_file, correlate_command] + corr_atom, stdout=log_file, stderr=err_file, cwd=working_dir))
                else:
                    processes.add(subprocess.Popen([sys.executable, base_command, in_file], stdout=log_file, stderr=err_file, cwd=working_dir))
            else:
                processes.add(subprocess.Popen([base_command, in_file], stdout=log_file, stderr=err_file, cwd=working_dir))
            processes = wait(processes, threads, wait_time, logfile=logfile)

    wait(processes, 1, wait_time, logfile=logfile)
    csv_list = [infile.split('.in')[0] + '.batch.csv' for infile in all_in_files]
    return csv_list


def cleanup(output_path, no_compress=False):
    """ Used to clean up the large number of intermediate files generated by fortran isodist. Moves these to their
    own folder, and optionally, compresses them into a single .gz file.
    :param output_path: path to the output folder.
    :param no_compress: bool determining whether the files should be compressed.

    """
    base_name = output_path.split('/')[-1].split('_output.csv')[0]
    base_path = '/'.join(output_path.split('/')[:-1]) + '/'

    isodist_dirs = ['_isodist_inputs', '_isodist_fits', '_isodist_outputs', '_isodist_logs']
    for ex in isodist_dirs:
        try:
            os.mkdir(base_path + base_name + ex)
        except OSError:
            print(
                '...the output directory: ' + base_path + base_name + ex +
                ' already exists, and files within it may be overwritten. continue? [y/n]')
            choice = input().lower()
            if choice == 'y':
                shutil.rmtree(base_path + base_name + ex)
                os.mkdir(base_path + base_name + ex)
            else:
                raise

    in_files = glob.glob(base_path + '*.in*')
    batch_files = glob.glob(base_path + '*.batch')
    log_files = glob.glob(base_path + '*.log')
    err_files = glob.glob(base_path + '*.err')
    if no_compress:
        for f in in_files + batch_files:
            shutil.move(f, base_path + base_name + '_isodist_inputs')
        for f in log_files + err_files:
            shutil.move(f, base_path + base_name + '_isodist_logs')
    else:
        with tarfile.open(base_path + base_name + '_isodist_inputs/batch_files.tar.gz', 'x:gz') as tar:
            for i in batch_files:
                tar.add(i)
                os.remove(i)
        with tarfile.open(base_path + base_name + '_isodist_inputs/in_files.tar.gz', 'x:gz') as tar:
            for i in in_files:
                tar.add(i)
                os.remove(i)
        with tarfile.open(base_path + base_name + '_isodist_logs/log_files.tar.gz', 'x:gz') as tar:
            for i in log_files:
                tar.add(i)
                os.remove(i)
        with tarfile.open(base_path + base_name + '_isodist_logs/err_files.tar.gz', 'x:gz') as tar:
            for i in err_files:
                tar.add(i)
                os.remove(i)

    dat_files = glob.glob(base_path + '*spectra/*.dat')
    for i in dat_files:
        os.remove(i)

    fit_files = glob.glob(base_path + '*spectra/*.fit')
    for i in fit_files:
        shutil.move(i, base_path + base_name + '_isodist_fits')

    csv_files = glob.glob(base_path + '*.batch.csv')
    for i in csv_files:
        shutil.move(i, base_path + base_name + '_isodist_outputs')
    shutil.move(output_path, base_path + base_name + '_isodist_outputs')


def compile_isodist_csvs(csv_list, output_csv_name, parsed_pysodist_input=None, logfile=None):
    """ Function to compile and format the individual isodist csvs into a single complete csv file.
    :param csv_list: list of the full paths to the individual .csv files
    :param parsed_pysodist_input: tsv file that was used to extract spectra -
    useful in providing additional info such as protein name for each peptide.
    :param output_csv_name: string with the full path of the resulting .csv file
    (this will be compiled from the individual isodist runs)
    :param logfile: path to file to log

    :return: a pandas dataframe with all of the final fit params from all of the isodist runs.
    """

    pd_list = []
    if not (parsed_pysodist_input is None):
        parsed_tsv = pd.read_csv(parsed_pysodist_input, sep='\t')
        associate_proteins = True
    else:
        parsed_tsv = None
        associate_proteins = False

    for current_csv in csv_list:
        with open(current_csv, 'r') as file:
            fixed = file.read().replace(',\n', '\n')
        with open(current_csv, 'w') as file:
            file.write(fixed)
        parsed_csv = pd.read_csv(current_csv)
        try:
            parsed_csv = parsed_csv.drop(['tim', 'symb'], axis=1)
        except:
            pass

        if associate_proteins:
            for row in range(parsed_csv.shape[0]):
                current_peptide = parsed_csv.loc[row, 'pep']
                associated_protein = \
                    parsed_tsv[parsed_tsv['peptide_modified_sequence'] == current_peptide]['protein_IDs'].values
                assert associated_protein.shape[0] > 0, \
                    current_peptide + ' not found in pd_parsed_report, aborting. Check this file.'
                parsed_csv.loc[row, 'protein'] = associated_protein[0]
        pd_list.append(parsed_csv)

    compiled_pd = pd.concat(pd_list, ignore_index=True)
    compiled_pd.rename(columns=lambda x: x.strip(), inplace=True)
    compiled_pd.to_csv(output_csv_name, index=False)
    log('compiled ' + str(len(csv_list)) + ' batch csvs into: ' + output_csv_name, logfile)
    return compiled_pd


def write_batch(current_peptide, batch_path, spectra_string):
    """
    Writes a new entry to a batch file that can be read by default (fortran) isodist
    :param current_peptide: a pandas dataframe containing a single row of the parsed tsv file
    :param batch_path: the full path to the batch file to append
    :param spectra_string: string pointing to the relative path for the file containing the spectra

    returns: None
    """

    with open(batch_path, 'a+') as to_write:
        string = ' '.join([current_peptide['peptide_modified_sequence'],
                           str(current_peptide['charge']),
                           spectra_string])
        to_write.write(string + '\n')


def write_isodist_input(batch_file_path, atomfile, resfile, niter=5, sigma=100.0, B=1.0, offset=0.01, GW=0.030):
    """
    Writes an isodist input file
    :param batch_file_path: the path to the batchfile on which to base the input files
    :param atomfile: path to the atomfile required by isodist
    :param resfile: path to the resfile required by isodist
    :param niter: number of iterations isodist runs
    :param sigma: isodist sigma param
    :param B: isodist B param
    :param offset: isodist offset param
    :param GW: isodist gaussian width param

    returns: None
    """
    batch_file_name = batch_file_path.split('/')[-1]
    in_file_path = batch_file_path.split('.batch')[0] + '.in'
    with open(in_file_path, 'w') as output:
        output.write('fitit = program options: fitit tryit\n')
        output.write('./' + batch_file_name + ' = batchfile: file containing peptides, chgs, peaks\n')
        output.write(atomfile + ' = atomfile\n')
        output.write(resfile + ' = resfile\n')
        output.write(str(niter) + ' = niter # of interactions for each round of least squares.\n')
        output.write(str(sigma) + ' sigma std deviation of noise (currently read by not used).\n')
        output.write(str(B) + ' auto = B : initial guess for baseline.\n')
        output.write(str(offset) + ' = OFF : initial guess for accuracy offset.\n')
        output.write(str(GW) + ' = GW : initial guess for gaussian width.\n')
    return in_file_path


def write_batch_files(batch_df, batch_base_path, batch_size=300, logfile=None):
    """
    Write the appropriate batch file(s)

    :param batch_df: the full dataframe of the spectra to be fit
    :param batch_base_path: string with the full path to the directory where the batches will be written.
    :param batch_size: the number of peptides to put into a single batch for isodist
    :param logfile: path to file to log

    returns: a list of pairs with the full path, and t.
    """

    if os.path.exists(batch_base_path + 'batch_0.batch'):
        print(batch_base_path + '_0.batch already exists. Proceeding will delete this file and start fresh.')
        print("Proceed? [y/n]")
        choice = input().lower()
        if choice == 'y':
            all_files = glob.glob(batch_base_path + 'batch_*.*')
            for file in all_files:
                os.remove(file)
    assert (os.path.exists(batch_base_path + 'batch_0.batch') is False)
    required_batches = batch_df.shape[0] // batch_size + 1
    log('there are ' + str(batch_df.shape[0]) + ' total spectra to fit...', logfile)
    log('writing ' + str(required_batches) + ' batch files...', logfile)

    written_list = []

    for index in range(batch_df.shape[0]):
        current_peptide = batch_df.iloc[index]
        batch_file_path = batch_base_path + 'batch_' + str(index // int(batch_size)) + '.batch'
        write_batch(current_peptide, batch_file_path, current_peptide['spectra_file'])
        written_list.append(batch_file_path)
    return sorted(set(written_list))


def prep_model_files(destination_path, atomfile_path, resfile_path):
    model_dir = destination_path + '/model_files/'
    try:
        os.mkdir(model_dir)
    except OSError:
        print(
            '...the model files directory: ' + model_dir + ' already exists, and files within it may be overwritten. '
                                                           'continue? [y/n]')
        choice = input().lower()
        if not choice == 'y':
            raise
    shutil.copy2(atomfile_path, model_dir + atomfile_path.split('/')[-1])
    shutil.copy2(resfile_path, model_dir + resfile_path.split('/')[-1])


def add_args(parser):
    parser.add_argument('input_file',
                        help='Path to the pd_exported_peaks.tsv (produced by extract_spectra.py). '
                             'This file should be in a folder containing a spectra folder that holds .tsvs to fit.')
    parser.add_argument('isodist_executable', nargs='?', default='~/software/pysodist/fortran/isodist',
                        help='Exact fortran command to execute. '
                             'e.g. ~/software/pysodist/fortran/isodist or C:/isodist_win/isodist_win.exe')
    parser.add_argument('atomfile', nargs='?', default='~/software/pysodist/model_files/atoms.txt',
                        help='Path to the atom definition file (e.g. exp_atom_defs.txt). '
                             'You will likely not need to modify this file.')
    parser.add_argument('modelfile', nargs='?', default='~/software/pysodist/model_files/U_var500N_fix998N.txt',
                        help='Path to the labeling file - you will need to edit this file based on your labeling'
                             'scheme. Your output uses this filename to associate model file used with output.')
    parser.add_argument('--config_file', default=None, help='Path to an optional configuration file (generated by'
                                                            'pysodist config). If provided, the isodist_command, '
                                                            'atomfile, resfile, and resolution params will be ignored.')
    parser.add_argument('--threads', default=2, type=int,
                        help='number of threads to use. typically 1 less than the number of cores available. Default=2')
    parser.add_argument('--wait_time', default=60, type=int,
                        help='Seconds between each test of whether isodist run has finished. Default=60 seconds')
    parser.add_argument('--pysodist_input', default=None,
                        help='Path to the "pd_parsed_report.tsv" file produced by parse_input. '
                             'Typically a .tsv file that has extracted the relevant info from a Skyline report. '
                             'Useful in providing additional info such as protein name for each peptide.')
    parser.add_argument('--no_cleanup', action='store_const', const=True, default=False,
                        help='Do not clean up the folder my moving around and deleting the intermediate files.')
    parser.add_argument('--no_compress', action='store_const', const=True, default=False,
                        help='Do not compress isodist log and intermediate processing files.')
    parser.add_argument('--logfile', default=None, help='Optionally provide a path to a logfile to store outputs')
    parser.add_argument('--pythonic', action='store_true', default=False, help='Analyze using the pythonic pipeline')
    parser.add_argument('--correlate', nargs = '+', default=None, help='Names of atoms in the residue model that should be considered as correlated')
    return parser


def main(args):
    logfile = args.logfile
    log('\n****INITIATING****', logfile)
    log('executed command: ' + " ".join(sys.argv), logfile)

    input_file = args.input_file.replace('\\', '/')
    log('checking input parameters', logfile)
    if not (args.config_file is None):
        config_file = args.config_file.replace('\\', '/')
        assert os.path.exists(config_file), 'The provided pysodist_config file does not exist. ' \
                                            'Please check that the path provided is correct.'
        config_df = pd.read_csv(config_file, sep='\t', index_col='FIELD')
        log('using configuration file: ' + config_file, logfile)
        atomfile = config_df.loc['atomfile']['VALUE'].replace('\\', '/')
        modelfile = config_df.loc['modelfile']['VALUE'].replace('\\', '/')
        isodist_executable = config_df.loc['isodist_executable']['VALUE'].replace('\\', '/')
    else:
        atomfile = args.atomfile.replace('\\', '/')
        modelfile = args.modelfile.replace('\\', '/')
        isodist_executable = args.isodist_executable.replace('\\', '/')

    assert os.path.exists(input_file), \
        'The provided input_file: ' + input_file + ' does not exist. Please check that the path provided is correct.'
    assert os.path.exists(atomfile), \
        'The provided atomfile: ' + atomfile + ' does not exist. Please check that the path provided is correct.'
    assert os.path.exists(modelfile), \
        'The provided modelfile: ' + modelfile + ' does not exist. Please check that the path is correct.'
    assert os.path.exists(isodist_executable), \
        'The provided isodist_executable: ' + isodist_executable + ' does not exist. Check the path.'
    if not (args.pysodist_input is None):
        assert os.path.exists(args.pysodist_input), \
            'The provided pysodist input file: ' + args.pysodist_input + ' does not exist. Check the path.'

    working_dir = '/'.join(input_file.split('/')[:-1])
    modelfile_name = modelfile.split('/')[-1].split('.txt')[0]
    output = working_dir + '/' + modelfile_name + '_output.csv'

    log('working in directory: ' + working_dir, logfile)
    prep_model_files(working_dir, atomfile, modelfile)

    batch_base_path = '/'.join(input_file.split('/')[:-1]) + '/'
    batch_df = pd.read_csv(input_file, sep='\t')
    num_spectra = batch_df.shape[0]
    batch_size = math.ceil(num_spectra / args.threads)
    batch_file_path_list = write_batch_files(batch_df, batch_base_path, batch_size=batch_size, logfile=logfile)

    in_file_list = []
    for batch_file_path in batch_file_path_list:
        batch_file_path = batch_file_path.replace('\\', '/')
        in_file_list.append(write_isodist_input(batch_file_path, atomfile, modelfile))

    csv_list = run_fortran_isodist(in_file_list, isodist_executable,
                                   threads=args.threads, wait_time=args.wait_time, logfile=logfile, pythonic = args.pythonic, corr_atom = args.correlate)
    compile_isodist_csvs(csv_list, output, parsed_pysodist_input=args.pysodist_input, logfile=logfile)
    if args.no_cleanup is False:
        log('cleaning up...', logfile)
        cleanup(output, no_compress=args.no_compress)
    log('\n++++COMPLETED run_isodist++++\n', logfile)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Fortran pysodist runner. '
                    'Used to run fortran version of isodist and to clean up the outputs for subsequent plotting.')
    add_args(argparser)
    main(argparser.parse_args())
