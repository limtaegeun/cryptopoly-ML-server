#!/bin/sh

# shellcheck disable=SC2039
#source venv/bin/activate
source ../trainCryptopoly/venv/bin/activate
gunicorn app:app