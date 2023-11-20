# -*- coding: utf-8 -*-
"""
@author: Joey Davis <jhdavis@mit.edu> jhdavislab.org
@author: Laurel Kinman
@version: 0.0.4
"""

import pysodist.commands.parse_input as parse_input
import pysodist.commands.extract_spectra as extract_spectra
import pysodist.commands.run_isodist as run_isodist
import pysodist.commands.plot_spectra as plot_spectra
import pysodist.commands.gaussian_filter as gaussian_filter
from pysodist.utils import utilities
import pysodist
import os
import argparse
import glob
import pandas as pd
import math
import subprocess


def add_args(parser):
    # required
    parser.add_argument('input', help='input file to parse. Currently only skyline report files are supported')
    parser.add_argument('mzml', help='the relative path to mzml file to be analyzed. For Thermo instruments, '
                                     'one should generate the .mzml file from the original .raw file using msconvert '
                                     'as follows: \ .\msconvert.exe ".\[YOUR_RAW_FILE].raw" -o "./" --mzML --64 -v '
                                     'mz64 --inten32 --noindex --filter "msLevel 1" --zlib')
    parser.add_argument('sample_name', help='name of the sample within the skyline report to be analyzed.')
    parser.add_argument('isodist_command', help='exact fortran command to execute. e.g. C:\isodist\isodist.exe')
    
    #optional arguments
    parser.add_argument('--test_peptide_list', default = None, help = 'list of peptides to use in testing pysodist '
                            'before full run; if provided, pysodist will run through test pipeline rather than full')

    # isodist/pysodist specific 
    parser.add_argument('--threads', default=4, type=int,
                        help='number of threads to use. typically 1 less than the number of cores available. Default=4')
    parser.add_argument('--wait_time', default=60, type=int,
                        help='number of seconds to wait between each polling to test if the isodist run has finished. '
                             'Default=60 seconds')
    parser.add_argument('--correlate', nargs = '+', default=None, help='Names of atoms in the residue model that should be considered as correlated')
    parser.add_argument('--atomfile', default = None, help = 'Path to atom file if overriding auto-generated')
    parser.add_argument('--resfile', default = None, help = 'Path to res file if overriding auto-generated')

    # extract spectra specific
    parser.add_argument('--output_directory', default='./',
                        help='Output files will be saved in this folder: 1 directory per sample in the skyline '
                             'report. Default = ./')
    parser.add_argument('--protein_list', default=None,
                        help='An optional list of the proteins to parse. By default, all proteins in the report are '
                             'analyzed.')
    parser.add_argument('--isotope', default='light',
                        help='Be default, it is assumed that the report contains a light isotope (no special '
                             'labeling), if this field is not present in the report, you can specify a different '
                             'field here (e.g. "heavy")')
    parser.add_argument('--q_value', default=0.00,
                        help='Used to optionally filter the report file based on the q_value. By default, no q_value '
                             'filtering is used.')
    parser.add_argument('--labeling', default='N15',
                        help='The labeling scheme used for the highest mass isotope envelope you expect to fit. E.g. '
                             'N15 or C13')
    parser.add_argument('--interp_only', action='store_const', const=True, default=False,
                        help='Only save the interpolated spectra instead of the raw spectra')
    parser.add_argument('--sum_only', action='store_const', const=True, default=False,
                        help='Only save summed (and interpolated) spectra instead of all individual spectra. Results '
                             'in 1 spectra per peptide')
    parser.add_argument('--interp_res', default=0.001, type=float,
                        help='Set the interpolation delta m/z - typical values from 0.01 to 0.001')

    # plotting specific
    parser.add_argument('--numerator', nargs='+', default=['AMP_U'],
                        help='list of the fields to use in the numerator of the abundance ratio calculation ('
                             'typically AMP_U, AMP_L, AMP_F, or some combination. Default is AMP_U.')
    parser.add_argument('--denominator', nargs='+', default=['AMP_U', 'AMP_F'],
                        help='list of the fields to use in the denominator of the abundance ratio calculation ('
                             'typically AMP_U, AMP_L, AMP_F, or some combination. Default is AMP_U, AMP_F')
    parser.add_argument('--no_png', action='store_const', const=True, default=False,
                        help='By default .png files for the plots will be saved. This option forces these to not be '
                             'saved.')
    parser.add_argument('--no_pdf', action='store_const', const=True, default=False,
                        help='By default .pdf files for the plots will be saved. This option forces these to not be '
                             'saved.')

    # filtering specific
    parser.add_argument('--spectrassr', type = float, default = 10, help = 'Spectra SSR to filter scans by')
    parser.add_argument('--ssr', type = float, default = 1, help = 'Gaussian fitting SSR to filter peptides by')
    parser.add_argument('--filt', type = str, default = 'AMP_F', help = 'Value to fit and filter by (AMP_U, AMP_L, AMP_F, or sum). Default is AMP_F')
    parser.add_argument('--plotfilt', type = bool, default = True, help = 'Whether to plot retained and filtered-out spectra')

    return parser


def main(args):
    assert args.isodist_command.endswith('isodist') or args.isodist_command.endswith('pysodist_v2.py'), 'Check input isodist_command'

    if args.isodist_command.endswith('isodist'):
        pythonic = False
    elif args.isodist_command.endswith('pysodist_v2.py'):
        pythonic = True

    sample_name = args.sample_name.strip()
    output_directory = args.output_directory.replace('\\', '/')
    output_directory = utilities.check_dir(output_directory, make = True)
    parsed_mzml = extract_spectra.parse_mzml(args.mzml)
    isodist_command = args.isodist_command.replace('\\', '/')

    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    if args.test_peptide_list:
        print('running parameter-testing pipeline on selected peptides')
    
        print('***stage 1: parsing input file...***')
        isodist_df = parse_input.parse_skyline(args.input, peptide_list = args.test_peptide_list, sample_list=[sample_name],
                                            isotope=args.isotope, q_value=args.q_value,
                                            output_directory=output_directory)[0]
        print('unique peptides found: ' + str(isodist_df.shape[0]))
        
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***stage 2: extracting spectra...***')
        parsed_report = output_directory + args.sample_name + '/pd_parsed_report.tsv'
        print('using pysodist report file: ' + parsed_report)
        assert (os.path.exists(parsed_report) is True)
        
        sample_output_directory = output_directory + args.sample_name + '/'
        assert (os.path.exists(sample_output_directory) is True)
        extract_spectra.extract_spectra(parsed_mzml, parsed_report, sample_output_directory,
                                        labeling=args.labeling, save_interp_spectra=args.interp_only,
                                        interp_res=args.interp_res, sum_spectra_only=args.sum_only)
        
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***stage 3: running interactive fitter...***')
        ###INSERT CODE HERE


        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***stage 3: running pysodist on a random batch...***')
        batch_size = 100
        isodist_df = parse_input.parse_skyline(args.input, protein_list=args.protein_list, peptide_list = None, sample_list=[sample_name],
                                            isotope=args.isotope, q_value=args.q_value,
                                            output_directory=output_directory, batch_size = batch_size)[0]

        parsed_report = output_directory + args.sample_name + '/pd_parsed_report_batch0.tsv'
        extract_spectra.extract_spectra(parsed_mzml, parsed_report, sample_output_directory,
                                        labeling=args.labeling, save_interp_spectra=args.interp_only,
                                        interp_res=args.interp_res, sum_spectra_only=args.sum_only)
        isodist_input_file = sample_output_directory + 'pd_exported_peaks.tsv'
        if args.atomfile:
            atomfile = args.atomfile.replace('\\', '/') 
        '''
        else:
            atomfile = ###FIX TO REFERENCE AUTO-GENERATED ATOM FILE
        '''
        if args.resfile:
            resfile = args.resfile.replace('\\', '/') 
        '''    
        else:
            resfile = ###FIX TO REFERENCE AUTO-GENERATED RES FILE
        '''
        resfile_name = resfile.split('/')[-1].split('.txt')[0]
        isodist_output_csv = sample_output_directory + resfile_name + '_output.csv'
        run_isodist.prep_model_files(sample_output_directory, atomfile, resfile) 
        
        batch_base_path = '/'.join(isodist_input_file.split('/')[:-1]) + '/'
        batch_df = pd.read_csv(isodist_input_file, sep='\t')
        num_spectra = batch_df.shape[0]
        batch_size = math.ceil(num_spectra / args.threads)
        batch_file_path_list = run_isodist.write_batch_files(batch_df, batch_base_path, batch_size=batch_size)

        in_file_list = []
        for batch_file_path in batch_file_path_list:
            batch_file_path = batch_file_path.replace('\\', '/')
            in_file_list.append(run_isodist.write_isodist_input(batch_file_path, atomfile, resfile))

        csv_list = run_isodist.run_fortran_isodist(in_file_list, isodist_command, threads=args.threads, wait_time=args.wait_time, pythonic=pythonic, corr_atom = args.correlate)
        run_isodist.compile_isodist_csvs(csv_list, isodist_output_csv, parsed_pysodist_input=parsed_report)
        

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***stage 4: filtering results...***')
        run_isodist.cleanup(isodist_output_csv)
        gaussian_filter.run_filtration(args.spectrassr, args.ssr, args.filt, output_directory, args.threads, subset = False, plotfilt = True)

        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('cleaning up...')
        run_isodist.cleanup_spectra(isodist_output_csv)
        
    else:
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***stage 1: parsing input file...***')
        sample_name = args.sample_name.strip()
        batch_size = 100
        isodist_df = parse_input.parse_skyline(args.input, protein_list=args.protein_list, sample_list=[sample_name],
                                            isotope=args.isotope, q_value=args.q_value,
                                            output_directory=output_directory, batch_size = batch_size)[0]
        print('unique peptides found: ' + str(isodist_df.shape[0]))
        
        print('iterating through batches')
        sample_output_directories = []
        pep_batch_list = glob.glob(output_directory + args.sample_name + '/pd_parsed_report*.tsv')
        for parsed_report in pep_batch_list:
            batch_num = parsed_report.split('parsed_report_batch')[-1].split('.tsv')[0]
            print(f'working on batch {batch_num}/{str(len(pep_batch_list))}')

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('***stage 2: extracting spectra...***')
            sample_output_directories.append(utilities.check_dir(output_directory + args.sample_name + f'/pep_batch{batch_num}', make = True))
            extract_spectra.extract_spectra(parsed_mzml, parsed_report, sample_output_directories[-1],
                                            labeling=args.labeling, save_interp_spectra=args.interp_only,
                                            interp_res=args.interp_res, sum_spectra_only=args.sum_only)
            
            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('***stage 3: fitting spectra using pysodist...***')
            isodist_input_file = sample_output_directories[-1] + 'pd_exported_peaks.tsv'
            
            if args.atomfile:
                atomfile = args.atomfile.replace('\\', '/')
            '''
            else:
                atomfile = ### FIX TO USE AUTO-GENERATED ATOMFILE
            '''
            if args.resfile:
                resfile = args.resfile.replace('\\', '/')
            '''
            else:
                resfile = ### FIX TO USE AUTO-GENERATED RESFILE
            '''
            resfile_name = resfile.split('/')[-1].split('.txt')[0]
            isodist_output_csv = sample_output_directories[-1] + resfile_name + '_output.csv'
            run_isodist.prep_model_files(sample_output_directories[-1], atomfile, resfile)

            batch_base_path = sample_output_directories[-1]
            batch_df = pd.read_csv(isodist_input_file, sep='\t')
            num_spectra = batch_df.shape[0]
            batch_size = math.ceil(num_spectra / args.threads)
            batch_file_path_list = run_isodist.write_batch_files(batch_df, batch_base_path, batch_size=batch_size)

            in_file_list = []
            for batch_file_path in batch_file_path_list:
                batch_file_path = batch_file_path.replace('\\', '/')
                in_file_list.append(run_isodist.write_isodist_input(batch_file_path, atomfile, resfile))

            csv_list = run_isodist.run_fortran_isodist(in_file_list, isodist_command, threads=args.threads, wait_time=args.wait_time, pythonic=pythonic, corr_atom = args.correlate)
            run_isodist.compile_isodist_csvs(csv_list, isodist_output_csv, parsed_pysodist_input=parsed_report)

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('***stage 4: filtering results...***')
            run_isodist.cleanup(isodist_output_csv)
            gaussian_filter.run_filtration(args.spectrassr, args.ssr, args.filt, sample_output_directories[-1], args.threads, subset = False, plotfilt = False)

            print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            print('cleaning up...')
            run_isodist.cleanup_spectra(isodist_output_csv)

        print('compiling filtered batch results')
        all_filtered = pd.DataFrame()
        for dir in sample_output_directories:
            filtered = pd.read_csv(dir + 'gaussian_fits/sub_pepscanfilt.csv', index_col = 0)
            all_filtered = pd.concat([all_filtered, filtered])
        
        all_filtered.to_csv('/'.join(sample_output_directories[-1].split('/')[:-2]) + 'filtered_results.csv')
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('***stage 5: generating plots of the results...***')
        #ONCE EVERYTHING ELSE IS RUNNING, DECIDE WHAT TO PLOT HERE

        print('copying analysis_template.ipynb jupyter notebook...')
        out_ipynb = output_directory + 'pysodist_analysis.ipynb'
        if not os.path.exists(out_ipynb):
            # noinspection PyProtectedMember
            root_path = pysodist._ROOT + '/'
            ipynb = root_path + 'utils/analysis_template.ipynb'
            cmd = f'cp {ipynb} {out_ipynb}'
            subprocess.check_call(cmd, shell=True)
        else:
            print(f'{out_ipynb} already exists. Skipping')



    return
    

    


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description='Pysodist runner - links together the various pysodist modules (parse_input, extract_spectra, '
                    'run_isodist, plot_spectra. Note that this is experimental and should only be used if you carefully'
                    'inspect the code for errors.')
    add_args(argparser)
    main(argparser.parse_args())