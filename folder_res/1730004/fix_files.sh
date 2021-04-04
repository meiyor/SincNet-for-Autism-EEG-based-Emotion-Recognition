dir="$1"
for f in "$dir"/*; do
  echo "$f"
  sal=$(cut -d'_' -f11 <<< "$f")
  if [[ $sal ]]; then
     number=$(cut -d'.' -f1 <<< "$sal")
     echo $number
     find "res_time_channel_14_test_data_sub_$2_$sal" -type f | xargs grep -n 'err_te' > "res_time_channel_14_test_data_sub_$2_{$number}_err_te.csv"
     find "res_time_channel_14_test_data_sub_$2_$sal" -type f | xargs grep -n 'train-epoch' > "res_time_channel_14_test_data_sub_$2_{$number}_train_epoch.csv"
     rm "res_time_channel_14_test_data_sub_$2_$sal"
  fi
  #name=$(cut -d'.' -f1 <<< "$sal")
  echo $sal
  #cp "${dir_s}/${name}.mat" $dir_d
done

