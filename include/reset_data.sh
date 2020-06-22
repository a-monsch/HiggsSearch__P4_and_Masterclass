#!/bin/sh

cd ../data

wget --max-redirect=20 -O for_analysis_download.zip https://www.dropbox.com/sh/3j648sojeimjmfh/AACeBAPUZkvsr0gHXULloRSWa?dl=0

unzip -o for_analysis_download.zip
rm for_analysis_download.zip

unzip -o for_long_analysis.zip
rm for_long_analysis.zip

unzip -o for_event_display.zip
rm for_event_display.zip

unzip -o for_widgets.zip
rm for_widgets.zip

cd ../notebooks
