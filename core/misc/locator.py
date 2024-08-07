from .metadata import Metadata
import pandas as pd 
from .helpers import fix_mun_psgc

CACHE = {}

class Locator: 
    def load_reference(context, admin_level):
        df = None 
        if context not in CACHE:
            filepath = f"./data/locations/{context}/metadata/{admin_level}.csv"
            df = pd.read_csv(filepath)
            indices = ["Ref. Id"]
            
            if "PSGC" in df: 
                indices.append("PSGC")
            if "Province" in df:
                indices.append("Province")
            if "Region" in df:
                indices.append("Region")
            if "Municity" in df:
                indices.append("Municity")
            if "Island Group" in df:
                indices.append("Island Group")
            if "District No." in df:
                indices.append("District No.")
            if "adm4_psgc" in df:
                indices.append("adm4_psgc")
            if "adm3_psgc" in df:
                indices.append("adm3_psgc")
            if "adm2_psgc" in df:
                indices.append("adm2_psgc")
            if "adm1_psgc" in df:
                indices.append("adm1_psgc")
            
            df.set_index(indices)

            CACHE[context] = df
        else:
            df = CACHE[context]
        return df

    def island_groups(
        context,
        name=None
    ): 
        locations = Locator.load_reference(context, "island-groups")
        
        return locations

    def regions(
        context,
        island_group=None,
        code=None,
        slug=None
    ): 
        locations = Locator.load_reference(context, "regions")

        if island_group:
            locations = locations.query(f"`Island Group` == '{island_group}'")

        if code: 
            locations = locations.query(f"`Ref. Id` == 'r:{code}'")
        elif code: 
            locations = locations.query(f"`Region` == '{name}'")

        return locations

    def provinces(
        context,
        island_group=None,
        region_name=None,
        region_code=None
    ): 
        locations = Locator.load_reference(context, "provinces")
        
        if island_group:
            locations = locations.query(f"`Island Group` == '{island_group}'")

        elif region_name:
            locations = locations.query(f"`Region.1` == '{region_name}'")

        elif region_code:
            regions = Locator.regions(context)
            region_name = str(
                regions
                    .query(f"`Ref. Id` == 'r:{region_code}'")
                    .iloc[0]['Region']
            ).strip()
            locations = locations.query(f"`Region.1` == '{region_name}'")

        return locations

    def districts(
        context,
        island_group=None,
        region_name=None,
        region_code=None,
        province_name=None,
        province_code=None
    ): 
        locations = Locator.load_reference(context, "districts")
      
        if island_group:
            regions = Locator.regions(context)
            ig_regions = list(
                regions
                    .query(f"`Island Group` == '{island_group}'")
                    ["Region (Abbr)"]
            )
            ig_regions = [x.replace("Region ", "") for x in ig_regions]
            locations = locations[locations["Region"].isin(ig_regions)]
        
        elif region_name:
            regions = Locator.regions(context)
            region_abbr = regions.query(f"Region == '{region_name}'")
            region_abbr = list(region["Region (Abbr)"])[0].replace("Region ", "")
            locations = locations.query(f"`Region` == '{region_abbr}'")

        elif region_code:
            regions = Locator.regions(context)
            region_abbr = regions.query(f"`Ref. Id` == 'r:{region_code}'")
            region_abbr  = list(region["Region (Abbr)"])[0].replace("Region ", "")
            locations = locations.query(f"`Region` == '{region_abbr}'")

        elif province_name:
            locations = locations.query(f"`Province` == '{province_name}'")

        elif province_code:
            provinces = Locator.provinces(context)
            province_name = provinces.query(f"`Ref. Id` == 'p:{province_code}'")
            province_name = list(province_name["Province"])[0]
            locations = locations.query(f"`Province` == '{province_name}'")
        
        return locations

    def municities(
        context,
        island_group=None,
        region_name=None,
        region_code=None,
        province_name=None,
        province_code=None,
        district_slug=None, 
        district_code=None,
        district_code_slug=None
    ): 
        locations = Locator.load_reference(context, "municities")
       
        if island_group:
            regions = Locator.regions(context)
            ig_regions = list(
                regions
                    .query(f"`Island Group` == '{island_group}'")
                    ["Region"]
            )
            locations = locations[locations["Region"].isin(ig_regions)]
        
        elif region_name:
            regions = Locator.regions(context)
            region = regions.query(f"Region == '{region_name}'")
            region_name = list(region["Region"])[0]
            locations = locations.query(f"`Region` == '{region_name}'")

        elif region_code:
            regions = Locator.regions(context)
            region_abbr = regions.query(f"`Ref. Id` == 'p:{region_code}'")
            region_abbr = list(region_abbr["Region"])[0]
            locations = locations.query(f"`Region` == '{region_abbr}'")

        elif province_name:
            locations = locations.query(f"`Province` == '{province_name}'")

        elif province_code:
            provinces = Locator.provinces(context)
            province_name = provinces.query(f"`Ref. Id` == 'p:{province_code}'")
            province_name = list(province_name["Province"])[0]
            locations = locations.query(f"`Province` == '{province_name}'")
        
        elif province_code:
            provinces = Locator.provinces(context)
            province_name = provinces.query(f"`Ref. Id` == 'p:{province_code}'")
            province_name = list(province_name["Province"])[0]
            locations = locations.query(f"`Province` == '{province_name}'")
        
        elif district_slug: 
            tokens = district_slug.split("|")
            locations = locations.query(
                f"`Province` == '{tokens[0]}'"
            )
            locations = locations.query(
                f"`District` == '{tokens[1]}'"
            )           
            
        elif district_code_slug: 
            tokens = district_code_slug.split("|")
            
            province_code = tokens[0]
            provinces = Locator.provinces(context)
            province_name = provinces.query(f"`Ref. Id` == 'p:{province_code}'")
            province_name = list(province_name["Province"])[0]

            locations = locations.query(
                f"`Province` == '{province_name}'"
            )
            locations = locations.query(
                f"`District` == '{tokens[1]}'"
            )

        return locations

    def barangays(
        context,
        island_group=None,
        region_name=None,
        region_code=None,
        province_name=None,
        province_code=None,
        district_slug=None, 
        district_code=None,
        district_code_slug=None,
        municity_slug=None,
        municity_code_slug=None
    ): 
        locations = Locator.load_reference(context, "barangays")

        if island_group:
            regions = Locator.regions(context)
            psgcs = list(
                regions
                    .query(f"`Island Group` == '{island_group}'")
                    ["Ref. Id"]
            )
            psgcs = [int(x.split(":")[1]) for x in psgcs]
            locations = locations[locations["adm1_psgc"].isin(psgcs)]

        elif region_name:
            regions = Locator.regions(context)
            psgcs = list(
                regions
                    .query(f"`Region` == '{region_name}'")
                    ["Ref. Id"]
            )
            psgcs = [int(x.split(":")[1]) for x in psgcs]
            locations = locations[locations["adm1_psgc"].isin(psgcs)]

        elif region_code:
            regions = Locator.regions(context)
            locations = locations[
                locations["adm1_psgc"].isin([int(region_code)])
            ]

        elif province_name:
            provinces = Locator.provinces(context)
            psgcs = list(
                provinces
                    .query(f"`Province` == '{province_name}'")
                    ["Ref. Id"]
            )
            psgcs = [int(x.split(":")[1]) for x in psgcs]
            locations = locations[locations["adm2_psgc"].isin(psgcs)]

        elif province_code:
            regions = Locator.provinces(context)
            locations = locations[
                locations["adm2_psgc"].isin([int(province_code)])
            ]
        
        elif district_slug:
            tokens = district_slug.split("|")
            municities = Locator.municities(context)
            municities = municities.query(f"`Province` == '{tokens[0]}'")
            municities = municities.query(f"`District` == '{tokens[1]}'")
            municities = list(municities["Ref. Id"])
            psgcs = [x.split(":")[1] for x in municities]
            psgcs = [int(fix_mun_psgc(x)) for x in psgcs]
            locations = locations[
                locations["adm3_psgc"].isin(psgcs)
            ]

        elif district_code_slug:
            tokens = district_code_slug.split("|")
            provinces = Locator.provinces(context)
            province_name = provinces.query(f"`Ref. Id` == 'p:{tokens[0]}'")
            province_name = list(province_name["Province"])[0]
            municities = Locator.municities(context)
            municities = municities.query(f"`Province` == '{province_name}'")
            municities = municities.query(f"`District` == '{tokens[1]}'")
            municities = list(municities["Ref. Id"])
            psgcs = [x.split(":")[1] for x in municities]
            psgcs = [int(fix_mun_psgc(x)) for x in psgcs]
            locations = locations[
                locations["adm3_psgc"].isin(psgcs)
            ]

        elif municity_slug:
            tokens = municity_slug.split("|")
            municities = Locator.municities(context)
            municities = municities.query(f"`Province` == '{tokens[0]}'")
            municities = municities.query(f"`Municity` == '{tokens[1]}'")
            municities = list(municities["Ref. Id"])
            psgcs = [x.split(":")[1] for x in municities]
            psgcs = [int(fix_mun_psgc(x)) for x in psgcs]
            locations = locations.query(f"adm3_psgc == {psgcs[0]}")

        elif municity_code_slug:
            tokens = municity_code_slug.split("|")
            provinces = Locator.provinces(context)
            province_name = provinces.query(f"`Ref. Id` == 'p:{tokens[0]}'")
            province_name = list(province_name["Province"])[0]
            municities = Locator.municities(context)
            municities = municities.query(f"`Province` == '{province_name}'")
            municities = municities.query(f"`Municity` == '{tokens[1]}'")
            municities = list(municities["Ref. Id"])
            psgcs = [x.split(":")[1] for x in municities]
            psgcs = [int(fix_mun_psgc(x)) for x in psgcs]
            locations = locations[
                locations["adm3_psgc"].isin(psgcs)
            ]

        return locations



    def island_group(context, name=None, code=None): 
        locations = Locator.island_groups(context)
        result = None

        if name: 
            result = locations.query(f"`Island Group` == '{name}'")

        return result

    def region(context, name=None, code=None): 
        locations = Locator.regions(context)
        result = None

        if name: 
            result = locations.query(f"`Region` == '{name}'")
        if code: 
            result = locations.query(f"`Ref. Id` == 'r:{code}'")

        return result

    def province(context, name=None, code=None): 
        locations = Locator.provinces(context)
        result = None

        if name: 
            result = locations.query(f"`Province` == '{name}'")
        if code: 
            result = locations.query(f"`Ref. Id` == 'p:{code}'")

        return result

    def district(context, slug=None, code_slug=None): 
        locations = Locator.districts(context)
        result = None
        
        if slug: 
            tokens = slug.split("|")
            result = locations.query(f"`Province` == '{tokens[0]}'")
            result = result.query(f"`District No.` == '{int(tokens[1])}'")
            return result

        return result

    def municity(context, slug=None, code=None): 
        locations = Locator.municities(context)
        result = None
        
        if code:
            result = locations.query(f"`Ref. Id` == 'm:{code}'")
        if slug:
            tokens = slug.split("|")
            locations = locations.query(f"`Province` == '{tokens[0]}'")
            locations = locations.query(f"`Municity` == '{tokens[1]}'")
            result = locations

        return result

    def barangay(context, slug=None, code=None): 
        locations = Locator.barangays(context)
        result = None
        
        if code:
            result = locations.query(f"adm4_psgc == {int(code)}")
        elif slug: 
            tokens = slug.split("|")
            provinces = Locator.provinces(context)
            province = provinces.query(f"`Province` == '{tokens[0]}'")
            adm2_code = list(province["PSGC"])[0]
            municities = Locator.municities(context)
            municity = municities.query(f"`Province` == '{tokens[0]}'")
            municity = municity.query(f"`Municity` == '{tokens[1]}'")
            adm3_code = str(list(municity["PSGC"])[0])
            adm3_code = str(fix_mun_psgc(adm3_code)).replace(".0", "")
            locations = locations.query(f"adm2_psgc == {adm2_code}")
            locations = locations.query(f"adm3_psgc == {adm3_code}")

        return result

    def locate(context, ref_id=None): 
        tokens = ref_id.split(":")
        if tokens[0] == "ig": 
            q = Locator.island_groups(context)
        elif tokens[0] == "r": 
            q = Locator.regions(context)
        elif tokens[0] == "p": 
            q = Locator.provinces(context)
        elif tokens[0] == "r": 
            q = Locator.regions(context)
        elif tokens[0] == "d": 
            q = Locator.districts(context)
        elif tokens[0] == "m": 
            q = Locator.municities(context)
        elif tokens[0] == "b": 
            q = Locator.barangays(context)
        q = q.query(f"`Ref. Id` == '{ref_id}'")
        return q

    def name(context, ref_id="q"):
        tokens = ref_id.split(":")
        q = Locator.locate(context, ref_id)
        if tokens[0] == "ig": 
            q = list(q["Island Group"])
        elif tokens[0] == "r": 
            q = list(q["Region"])
        elif tokens[0] == "p": 
            q = list(q["Province"])
        elif tokens[0] == "d": 
            q = list(q["District Key"])
        elif tokens[0] == "m": 
            q = list(q["Municity"])
        elif tokens[0] == "b": 
            q = list(q["adm4_en"])
        
        if len(q) == 0: 
            return None 
        else: 
            return q