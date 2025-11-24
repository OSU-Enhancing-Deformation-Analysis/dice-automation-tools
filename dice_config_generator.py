#!/usr/bin/env python3
"""
DICe Configuration Generator
Automatically scans image sequences and generates DICe configuration files.
"""
import argparse
from pathlib import Path
from PIL import Image
import json
import re

class DICeConfigGenerator:
    def __init__(self, image_dir, output_dir, subset_size=41, step_size=50):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.subset_size = subset_size
        self.step_size = step_size
        
    def scan_images(self):
        """Scan image sequence."""
        tif_files = sorted(list(self.image_dir.glob("*.tif")) + 
                          list(self.image_dir.glob("*.tiff")))
        
        if not tif_files:
            raise ValueError(f"No TIFF images found in {self.image_dir}")
        
        # Get image dimensions
        img = Image.open(tif_files[0])
        width, height = img.size
        
        # Detect file naming pattern
        first_name = tif_files[0].stem
        if first_name.isdigit():
            prefix = ""
            num_digits = len(first_name)
        else:
            # Extract numeric part
            match = re.search(r'(\d+)$', first_name)
            if match:
                prefix = first_name[:match.start()]
                num_digits = len(match.group(1))
            else:
                raise ValueError(f"Cannot detect numbering pattern in {first_name}")
        
        return {
            'num_images': len(tif_files),
            'width': width,
            'height': height,
            'extension': tif_files[0].suffix,
            'prefix': prefix,
            'num_digits': num_digits,
            'first_index': 1,
            'last_index': len(tif_files)
        }
    
    def generate_input_xml(self, info):
        """Generate input.xml."""
        xml_content = f"""<ParameterList>
  <Parameter name="output_folder" type="string" value="{self.output_dir}/"/>
  <Parameter name="image_folder" type="string" value="{self.image_dir}/"/>
  <Parameter name="image_file_prefix" type="string" value="{info['prefix']}"/>
  <Parameter name="image_file_extension" type="string" value="{info['extension']}"/>
  <Parameter name="reference_image_index" type="int" value="1"/>
  <Parameter name="start_image_index" type="int" value="2"/>
  <Parameter name="end_image_index" type="int" value="{info['last_index']}"/>
  <Parameter name="num_file_suffix_digits" type="int" value="{info['num_digits']}"/>
  <Parameter name="subset_size" type="int" value="{self.subset_size}"/>
  <Parameter name="step_size" type="int" value="{self.step_size}"/>
  <Parameter name="correlation_parameters_file" type="string" value="params.xml"/>
  <Parameter name="subset_file" type="string" value="subsets.txt"/>
</ParameterList>
"""
        output_path = self.output_dir / "input.xml"
        output_path.write_text(xml_content)
        return output_path
    
    def generate_params_xml(self):
        """Generate params.xml."""
        xml_content = """<ParameterList>
  <Parameter name="initialization_method" type="string" value="USE_FIELD_VALUES"/>
  <Parameter name="optimization_method" type="string" value="SIMPLEX"/>
  <Parameter name="interpolation_method" type="string" value="KEYS_FOURTH"/>
  <Parameter name="enable_translation" type="bool" value="true"/>
  <Parameter name="enable_rotation" type="bool" value="false"/>
  <Parameter name="enable_normal_strain" type="bool" value="true"/>
  <Parameter name="enable_shear_strain" type="bool" value="true"/>
  <Parameter name="output_delimiter" type="string" value=","/>
  
  <ParameterList name="post_process_vsg_strain">
    <Parameter name="strain_window_size_in_pixels" type="int" value="51"/>
  </ParameterList>
  
  <ParameterList name="output_spec">
    <Parameter name="COORDINATE_X" type="bool" value="true"/>
    <Parameter name="COORDINATE_Y" type="bool" value="true"/>
    <Parameter name="DISPLACEMENT_X" type="bool" value="true"/>
    <Parameter name="DISPLACEMENT_Y" type="bool" value="true"/>
    <Parameter name="VSG_STRAIN_XX" type="bool" value="true"/>
    <Parameter name="VSG_STRAIN_YY" type="bool" value="true"/>
    <Parameter name="VSG_STRAIN_XY" type="bool" value="true"/>
    <Parameter name="SIGMA" type="bool" value="true"/>
    <Parameter name="GAMMA" type="bool" value="true"/>
    <Parameter name="MATCH" type="bool" value="true"/>
  </ParameterList>
</ParameterList>
"""
        output_path = self.output_dir / "params.xml"
        output_path.write_text(xml_content)
        return output_path
    
    def generate_subsets_txt(self, info):
        """Generate subsets.txt."""
        # Calculate ROI boundaries (avoid edges)
        margin = 30
        center_x = info['width'] // 2
        center_y = info['height'] // 2
        width = info['width'] - 2 * margin
        height = info['height'] - 2 * margin
        
        txt_content = f"""BEGIN REGION_OF_INTEREST
  BEGIN BOUNDARY
    BEGIN RECTANGLE
      Center {center_x} {center_y}
      Width {width}
      Height {height}
    END RECTANGLE
  END BOUNDARY
END REGION_OF_INTEREST
"""
        output_path = self.output_dir / "subsets.txt"
        output_path.write_text(txt_content)
        return output_path
    
    def generate(self):
        """Generate all configuration files."""
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_dir = self.output_dir / "results"
        results_dir.mkdir(exist_ok=True)
        
        # Scan images
        print(f"Scanning images in {self.image_dir}...")
        info = self.scan_images()
        
        print(f"  Found {info['num_images']} images")
        print(f"  Image size: {info['width']}x{info['height']}")
        print(f"  File pattern: {info['prefix']}<number>{info['extension']}")
        
        # Generate configurations
        print(f"\nGenerating DICe configuration in {self.output_dir}...")
        input_xml = self.generate_input_xml(info)
        params_xml = self.generate_params_xml()
        subsets_txt = self.generate_subsets_txt(info)
        
        print(f"  Created {input_xml.name}")
        print(f"  Created {params_xml.name}")
        print(f"  Created {subsets_txt.name}")
        
        # Save sequence info
        info_json = self.output_dir / "sequence_info.json"
        with open(info_json, 'w') as f:
            json.dump(info, f, indent=2)
        print(f"  Created {info_json.name}")
        
        # Estimate processing time
        num_frames = info['last_index'] - 1
        num_points = (info['width'] // self.step_size) * (info['height'] // self.step_size)
        estimated_minutes = (num_frames * num_points * 0.026) / 60  # Based on test data
        
        print(f"\nConfiguration complete!")
        print(f"  Tracking points: ~{num_points}")
        print(f"  Frames to process: {num_frames}")
        print(f"  Estimated time: ~{estimated_minutes:.1f} minutes")
        
        return {
            'input_xml': str(input_xml),
            'params_xml': str(params_xml),
            'subsets_txt': str(subsets_txt),
            'sequence_info': info
        }

def main():
    parser = argparse.ArgumentParser(description='Generate DICe configuration files')
    parser.add_argument('image_dir', help='Directory containing TIFF image sequence')
    parser.add_argument('output_dir', help='Directory to write configuration files')
    parser.add_argument('--subset-size', type=int, default=41, 
                       help='Subset size in pixels (default: 41)')
    parser.add_argument('--step-size', type=int, default=50,
                       help='Step size between tracking points (default: 50)')
    
    args = parser.parse_args()
    
    generator = DICeConfigGenerator(
        args.image_dir,
        args.output_dir,
        subset_size=args.subset_size,
        step_size=args.step_size
    )
    
    try:
        generator.generate()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())